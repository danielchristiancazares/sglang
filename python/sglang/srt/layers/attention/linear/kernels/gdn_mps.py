from __future__ import annotations

import math

import torch

from sglang.srt.hardware_backend.mps.ops import gdn_decode, gdn_prefill
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


def _normalize_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key_dim = q.shape[-1]
    q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6) / key_dim
    k = (
        k
        * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1e-6)
        / math.sqrt(key_dim)
    )
    return q.contiguous(), k.contiguous()


class MpsGDNKernel(LinearAttnKernelBase):
    """Native Metal recurrent GDN decode with a correctness-first prefill."""

    supports_packed_decode = False

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return gdn_decode(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            ssm_states,
            cache_indices,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        outputs = gdn_prefill(
            q,
            k,
            v,
            g,
            beta,
            ssm_states,
            cache_indices,
            query_start_loc,
        )
        return outputs.unsqueeze(0), None, None

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        """Verify a linear MTP chain and retain every rollback checkpoint.

        NEXTN uses ``topk == 1``, so each request contributes one fixed-length
        chain.  Run the already validated native recurrent decode kernel once
        per draft position against a private state copy.  The persistent state
        remains untouched until the acceptance phase selects one checkpoint.
        """
        if retrieve_parent_token is not None:
            raise NotImplementedError(
                "Native MPS GDN verify currently supports linear draft chains only"
            )

        batch_size = query_start_loc.numel() - 1
        if batch_size <= 0 or q.shape[1] != batch_size * cache_steps:
            raise ValueError(
                "MPS GDN verify expects fixed request-major draft chains: "
                f"tokens={q.shape[1]}, batch={batch_size}, steps={cache_steps}"
            )

        state_slots = cache_indices[:batch_size].to(torch.long)
        scratch_rows = intermediate_state_indices[:batch_size].to(torch.long)
        working_state = ssm_states.index_select(0, state_slots).clone()
        local_slots = torch.arange(batch_size, dtype=torch.int32, device=q.device)

        q = q.reshape(1, batch_size, cache_steps, *q.shape[2:])
        k = k.reshape(1, batch_size, cache_steps, *k.shape[2:])
        v = v.reshape(1, batch_size, cache_steps, *v.shape[2:])
        a = a.reshape(batch_size, cache_steps, -1)
        b = b.reshape(batch_size, cache_steps, -1)

        outputs = []
        for step in range(cache_steps):
            output = gdn_decode(
                q[:, :, step],
                k[:, :, step],
                v[:, :, step],
                a[:, step],
                b[:, step],
                A_log,
                dt_bias,
                working_state,
                local_slots,
            )
            intermediate_states_buffer[scratch_rows, step] = working_state
            outputs.append(output.squeeze(0))

        return torch.stack(outputs, dim=1).reshape(1, -1, *v.shape[3:])
