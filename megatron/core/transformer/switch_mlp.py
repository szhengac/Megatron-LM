# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import (
    get_tensor_and_expert_parallel_group,
    get_tensor_model_parallel_group,
    get_expert_parallel_group,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name, all_to_all
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from .mlp import MLP, MLPSubmodules


_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()


def sinkhorn(cost, tol=0.0001, penalty=1.0):
    "Sinkhorn based MoE routing function"
    if penalty != 1.0:
        cost /= penalty
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)


def get_router_linear_layer(config):
    router = torch.nn.Linear(config.hidden_size, config.num_moe_experts, bias=False)
    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
        config.init_method(router.weight)
    setattr(router.weight, 'sequence_parallel', config.sequence_parallel)
    return router


def batched_load_balancing_loss(config):
    # tokens_per_expert[i].shape = (num_experts)
    # expert_scores[i].shape = (tokens, num_experts)
    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    num_layers_per_pipeline_stage = (
        config.num_layers // config.pipeline_model_parallel_size)
    if config.virtual_pipeline_model_parallel_size is not None:
        num_layers_per_pipeline_stage /= config.virtual_pipeline_model_parallel_size

    if len(tokens_per_expert) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} token_per_experts "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{config.num_layers}\npipeline_model_parallel_size = "
            f"{config.pipeline_model_parallel_size}\n"
            "virtual_pipeline_model_parallel_size"
            f" = {config.virtual_pipeline_model_parallel_size}")
    if len(expert_scores) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} expert_scores "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{config.num_layers}\npipeline_model_parallel_size = "
            f"{config.pipeline_model_parallel_size}\n"
            "virtual_pipeline_model_parallel_size"
            f" = {config.virtual_pipeline_model_parallel_size}")

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all([
        x.ndim == 1 and x.numel() == config.num_moe_experts
        for x in tokens_per_expert
    ])

    tokens = expert_scores[0].shape[0]
    assert all([
        (x.ndim == 2 and x.shape[1] == config.num_moe_experts and
         x.shape[0] == tokens) for x in expert_scores
    ])


    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

    expected_values = num_layers_per_pipeline_stage * config.num_moe_experts
    assert tokens_per_expert.numel() == expected_values
    assert expert_scores.numel() == expected_values

    # Calculate the total scale across all factors.
    #
    # loss_weight * num_experts / (num_layers * tokens * top_k)
    scale_numerator = (
        config.num_moe_experts *
        config.moe_loss_weight
    )
    scale_denominator = (
        config.num_layers *
        tokens *
        config.num_experts_per_token
    )
    scale = scale_numerator / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores)


class SwitchMLP(MegatronModule):
    """
    Top-1 Mixture of Experts Layer. Routes input to one of N MLP "experts"
    Curently supports Sinkhorn based expert routing.
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.router = get_router_linear_layer(self.config)
        self.add_bias = config.add_bias_linear
        self.sequence_parallel = config.sequence_parallel
        self.route_algo = sinkhorn
        self.router_activation = torch.softmax
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        assert self.config.num_experts_per_token <= self.config.num_moe_experts
        self.num_experts_per_token = self.config.num_experts_per_token
        self.num_moe_experts = self.config.num_moe_experts
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def gather_indices(self, local_indices):
        """ Gather tensors and concatenate along the first dimension."""
        group = get_tensor_and_expert_parallel_group()
        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return local_indices

        dim_size = list(local_indices.size())
        dim_size[0] = dim_size[0] * world_size

        # TODO pre allocate memory
        output = torch.empty(
            dim_size, dtype=local_indices.dtype, device=torch.cuda.current_device()
        )
        torch.distributed._all_gather_base(output, local_indices.contiguous(), group=group)
        return output

    def gather_forward(self, hidden_states, max_ind):
        if self.sequence_parallel or (self.expert_parallel_size > 1):
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            global_indices = self.gather_indices(max_ind)
        else:
            global_hidden_states = hidden_states
            global_indices = max_ind

        output_total = torch.zeros_like(global_hidden_states)
        output_bias_total = None
        if self.add_bias:
            output_bias_total = torch.zeros_like(global_hidden_states)

        for expert_num, expert in enumerate(self.local_experts):
            local_expert_index = self.local_expert_indices[expert_num]
            local_indices = (global_indices == local_expert_index).nonzero()
            hidden = global_hidden_states[local_indices, :]
            output, output_bias = expert(hidden)

            output_total[local_indices, :] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_total[local_indices, :] = output_bias

        if self.sequence_parallel or (self.expert_parallel_size > 1):
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                output_total
            )
            if self.add_bias:
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    output_bias_total
                )
                # bias is duplicated across tensor parallelism ranks for row parallel linear;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )

        return output_total, output_bias_total

    def alltoall_forward(self, hidden_states, max_ind):
        expert_group = get_expert_parallel_group()

        with torch.no_grad():
            sorted_max_ind, indices = torch.sort(max_ind, dim=0)
            bin_counts = torch.bincount(sorted_max_ind, minlength=self.num_moe_experts)
            routed_bin_counts = torch.zeros_like(bin_counts)
            torch.distributed.all_to_all_single(routed_bin_counts, bin_counts, group=expert_group)
            if self.num_local_experts > 1:
                routed_bin_counts_list = routed_bin_counts.reshape((-1, self.num_local_experts)).sum(dim=-1).tolist()
                bin_counts_list = bin_counts.reshape((-1, self.num_local_experts)).sum(dim=-1).tolist()
            else:
                routed_bin_counts_list = routed_bin_counts.tolist()
                bin_counts_list = bin_counts.tolist()
   
        hidden_states = hidden_states[indices.tolist(), ...]
        hidden_states, ep_handle = all_to_all(hidden_states, routed_bin_counts_list, bin_counts_list, expert_group, async_op=True)

        if self.num_local_experts > 1:
            if self.sequence_parallel:
                with torch.no_grad():
                    total_routed_bin_counts = tensor_parallel.gather_from_sequence_parallel_region(
                        routed_bin_counts
                    )
                    global_hidden_state_sizes = \
                            total_routed_bin_counts.reshape((get_tensor_model_parallel_world_size(), self.num_moe_experts)).sum(dim=-1).to(device='cpu', non_blocking=True)

            local_token_indices = \
                    torch.repeat_interleave(torch.arange(self.num_local_experts,
                        device=hidden_states.device).repeat(get_tensor_model_parallel_world_size() * self.expert_parallel_size),
                        total_routed_bin_counts)
            global_hidden_state_sizes = global_hidden_state_sizes.tolist()
        else:
            if self.sequence_parallel:
                with torch.no_grad():
                    total_routed_bin_counts = tensor_parallel.gather_from_sequence_parallel_region(
                        routed_bin_counts.sum(dim=0, keepdim=True)
                    )
                    global_hidden_state_sizes = total_routed_bin_counts.tolist()

        if self.sequence_parallel:
            ep_handle.wait()
            hidden_states = tensor_parallel.gather_from_uneven_sequence_parallel_region(
                hidden_states, global_hidden_state_sizes
            )

        output_bias_total = None
        if self.num_local_experts > 1:
            output_total = torch.zeros_like(hidden_states)
            if self.add_bias:
                output_bias_total = torch.zeros_like(hidden_states)
            for local_expert_index, expert in enumerate(self.local_experts):
                local_indices = (local_token_indices == local_expert_index).nonzero()
                hidden = hidden_states[local_indices, ...]
                output, output_bias = expert(hidden)
                output_total[local_indices, ...] = output
                if self.add_bias:
                    output_bias = output_bias.expand_as(output)
                    output_bias_total[local_indices, ...] = output_bias
        else:
            output, output_bias = self.local_experts[0](hidden_states.unsqueeze(1))
            output_total = output.squeeze(1)
            if self.add_bias:
                output_bias_total = output_bias.expand_as(output)

        if self.sequence_parallel:
            output_total = tensor_parallel.reduce_scatter_to_uneven_sequence_parallel_region(
                torch.split(output_total, global_hidden_state_sizes, dim=0)
            )
            if self.add_bias:
                output_bias_total = tensor_parallel.reduce_scatter_to_uneven_sequence_parallel_region(
                    torch.split(output_total_bias, global_hidden_state_sizes, dim=0)
                )
                # bias is duplicated across tensor parallelism ranks for row parallel linear;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / get_tensor_model_parallel_world_size()
                )

        output_total, ep_handle = all_to_all(output_total, bin_counts_list, routed_bin_counts_list, expert_group, async_op=True)
        if self.add_bias:
            output_total_bias, ep_bias_handle = all_to_all(output_total_bias, bin_counts_list, routed_bin_counts_list, expert_group, async_op=True)

        reversed_indices = torch.argsort(indices, dim=0).tolist()
        ep_handle.wait()
        output_total = output_total[reversed_indices, ...]
        if self.add_bias:
            ep_bias_handle.wait()
            output_total_bias = output_total_bias[reversed_indices, ...]

        return output_total, output_bias_total

    def token_shuffling(self, route):
        expert_group = get_expert_parallel_group()

        idx = torch.randperm(route.size()[0], device=route.device).tolist()
        shuffled_route = torch.empty_like(route)

        torch.distributed.all_to_all_single(shuffled_route, route[idx], group=expert_group)
        shuffled_norm_route = self.route_algo(
            shuffled_route.detach().to(dtype=torch.float32)
        )  # explicit fp32 conversion for stability
        _, shuffled_max_ind = torch.topk(shuffled_norm_route, self.num_experts_per_token, dim=1)

        shuffled_max_ind = shuffled_max_ind.to(dtype=torch.uint8)
        max_ind_out = torch.empty_like(shuffled_max_ind)
        torch.distributed.all_to_all_single(max_ind_out, shuffled_max_ind, group=expert_group)
        max_ind = torch.empty_like(max_ind_out, dtype=torch.int)
        max_ind[idx] = max_ind_out.int()
        
        return max_ind.view(-1)

    def compute_bins(self, route, max_ind=None):
        with torch.no_grad():
            if max_ind is None:
                _, max_ind = torch.topk(route, self.num_experts_per_token, dim=1)
            max_ind = max_ind.view(-1)
            bin_counts = torch.histc(max_ind, bins=self.num_moe_experts, min=0, max=self.num_moe_experts - 1)
        return bin_counts

    def forward(self, hidden_states, forward_only=False):
        hidden_shape = hidden_states.shape
        route = self.router(hidden_states)
        route = route.view(-1, self.num_moe_experts)

        if self.training:
            with torch.no_grad():
                max_ind = self.token_shuffling(route)
            route = self.router_activation(route.float(), dim=1).to(dtype=route.dtype)
            max_prob = route[torch.arange(route.size(0)).repeat_interleave(self.num_experts_per_token).tolist(), max_ind.tolist()]
            max_prob = max_prob.view(-1, self.num_experts_per_token)
        else:
            route = self.router_activation(route.float(), dim=1).to(dtype=route.dtype)
            max_prob, max_ind = torch.topk(route, self.num_experts_per_token, dim=1)
            max_ind = max_ind.view(-1)

        if self.num_experts_per_token > 1:
            max_prob /= max_prob.sum(dim=-1, keepdim=True)
        max_prob = max_prob.view(-1).unsqueeze(1)
        hidden_states = hidden_states.view(-1, hidden_shape[-1]).repeat_interleave(self.num_experts_per_token, dim=0)

        #output_total, output_bias_total = self.gather_forward(hidden_states, max_ind)
        output_total, output_bias_total = self.alltoall_forward(hidden_states, max_ind)

        output_total = output_total * max_prob
        output_total = output_total.view(*hidden_shape[:-1], self.num_experts_per_token, hidden_shape[-1]).sum(dim=-2)
        if self.add_bias:
            output_bias_total = output_bias_total * max_prob
            output_bias_total = output_bias_total.view(*hidden_shape[:-1], self.num_experts_per_token, hidden_shape[-1]).sum(dim=-2)

        if not forward_only:
            if self.training:
                tokens_per_expert = self.compute_bins(route)
            else:
                tokens_per_expert = self.compute_bins(route, max_ind)
            save_load_balancing_loss((tokens_per_expert, route))

        return output_total, output_bias_total
