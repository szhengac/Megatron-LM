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


def sinkhorn(cost, tol=0.0001):
    "Sinkhorn based MoE routing function"
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

    def gather_forward(self, hidden_states, route, max_prob, max_ind, hidden_shape):
        if self.sequence_parallel or (self.expert_parallel_size > 1):
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            global_indices = self.gather_indices(max_ind)
        else:
            global_hidden_states = hidden_states
            global_indices = max_ind

        output_total = torch.zeros_like(global_hidden_states)
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
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )

        output_total = output_total * max_prob
        output_total = output_total.view(hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total * max_prob
            output_bias_total = output_bias_total.view(hidden_shape)
        else:
            output_bias_total = None
        return output_total, output_bias_total

    def alltoall_forward(self, hidden_states, route, max_prob, max_ind, hidden_shape):
        expert_group = get_expert_parallel_group()

        with torch.no_grad():
            sorted_max_ind, indices = torch.sort(max_ind, dim=0)
            bin_counts = torch.bincount(sorted_max_ind, minlength=self.config.num_moe_experts)
            routed_bin_counts = torch.zeros_like(bin_counts)
            torch.distributed.all_to_all_single(routed_bin_counts, bin_counts, group=expert_group)
            if self.num_local_experts > 1:
                routed_bin_counts_list = routed_bin_counts.reshape((-1, self.num_local_experts)).sum(dim=-1).tolist()
                bin_counts_list = bin_counts.reshape((-1, self.num_local_experts)).sum(dim=-1).tolist()
            else:
                routed_bin_counts_list = routed_bin_counts.tolist()
                bin_counts_list = bin_counts.tolist()
   
        hidden_states = hidden_states[indices, ...]
        hidden_states, ep_handle = all_to_all(hidden_states, routed_bin_counts_list, bin_counts_list, expert_group, async_op=True)
        if 0 in routed_bin_counts_list:
            print(f"routed_bin_counts_list={routed_bin_counts_list}, hidden_states.shape={hidden_states.shape}")

        if self.num_local_experts > 1:
            if self.sequence_parallel:
                with torch.no_grad():
                    total_routed_bin_counts = tensor_parallel.gather_from_sequence_parallel_region(
                        routed_bin_counts
                    )
                    global_hidden_state_sizes = \
                            total_routed_bin_counts.reshape((get_tensor_model_parallel_world_size(), self.config.num_moe_experts)).sum(dim=-1).tolist()

            local_token_indices = \
                    torch.repeat_interleave(torch.arange(self.num_local_experts,
                        device=hidden_states.device).repeat(get_tensor_model_parallel_world_size() * self.expert_parallel_size),
                        total_routed_bin_counts)
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
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / get_tensor_model_parallel_world_size()
                )

        output_total, ep_handle = all_to_all(output_total, bin_counts_list, routed_bin_counts_list, expert_group, async_op=True)
        if self.add_bias:
            output_total_bias, ep_bias_handle = all_to_all(output_total_bias, bin_counts_list, routed_bin_counts_list, expert_group, async_op=True)

        reversed_indices = torch.argsort(indices, dim=0)
        ep_handle.wait()
        output_total = output_total[reversed_indices, ...]
        if self.add_bias:
            ep_bias_handle.wait()
            output_total_bias = output_total_bias[reversed_indices, ...]

        output_total = output_total * max_prob
        output_total = output_total.view(hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total * max_prob
            output_bias_total = output_bias_total.view(hidden_shape)
        else:
            output_bias_total = None
        return output_total, output_bias_total

    def forward(self, hidden_states):
        hidden_shape = hidden_states.shape
        route = self.router(hidden_states)
        route = route.view(-1, self.config.num_moe_experts)

        if self.training:
            with torch.no_grad():
                expert_group = get_expert_parallel_group()
                idx = torch.randperm(route.size()[0])
                shuffled_route = torch.empty_like(route)
                torch.distributed.all_to_all_single(shuffled_route, route[idx], group=expert_group)
                shuffled_norm_route = self.route_algo(
                    shuffled_route.detach().to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, shuffled_max_ind = torch.max(shuffled_norm_route, dim=1)
                shuffled_max_ind = shuffled_max_ind.to(dtype=torch.uint8)
                max_ind_out = torch.empty_like(shuffled_max_ind)
                torch.distributed.all_to_all_single(max_ind_out, shuffled_max_ind, group=expert_group)
                max_ind = torch.empty_like(max_ind_out, dtype=torch.int)
                max_ind[idx] = max_ind_out.int() 
            route = self.router_activation(route.float(), dim=1).to(dtype=route.dtype)
            max_prob = route[torch.arange(route.size(0)), max_ind]
        else:
            route = self.router_activation(route.float(), dim=1).to(dtype=route.dtype)
            max_prob, max_ind = torch.max(route, dim=1)

        max_prob = torch.unsqueeze(max_prob, 1)
        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        #output_total, output_bias_total = self.gather_forward(hidden_states, route, max_prob, max_ind, hidden_shape)
        output_total, output_bias_total = self.alltoall_forward(hidden_states, route, max_prob, max_ind, hidden_shape)

        return output_total, output_bias_total
