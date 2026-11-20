"""Diagnostic compatibility registration for the image's split FLA package.

Uses dispatcher registration, without replacing existing operators or editing
installed packages. Caller retains the returned Library for registration lifetime.
Not enabled in the production trainer yet.
"""
import torch
from fla_npu.ops import ascendc

SCHEMAS = {
    'npu_recompute_w_u_fwd': '(Tensor k, Tensor v, Tensor beta, Tensor A, int chunk_size, *, Tensor? g=None, Tensor? gk=None, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor)',
    'npu_chunk_gated_delta_rule_fwd_h': '(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool output_final_state=False, int? chunk_size=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool state_v_first=False) -> (Tensor, Tensor, Tensor?)',
    'npu_chunk_fwd_o': '(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None) -> Tensor',
    'npu_chunk_bwd_dv_local': '(Tensor q, Tensor k, Tensor d_o, Tensor g, float scale, int chunk_size, *, Tensor? g_gamma=None, Tensor? A=None, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> Tensor',
    'npu_chunk_gated_delta_rule_bwd_dhu': '(Tensor q, Tensor k, Tensor w, Tensor d_o, Tensor dv, float scale, int chunk_size, *, Tensor? g=None, Tensor? gK=None, Tensor? h0=None, Tensor? dht=None, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor?, Tensor)',
    'npu_chunk_bwd_dqkwg': '(Tensor q, Tensor k, Tensor v, Tensor g, Tensor h, Tensor dox, Tensor dh, Tensor dv, int chunk_size, *, int[]? cu_seqlens=None, int[]? chunk_indices=None, float? scale=None) -> (Tensor, Tensor, Tensor, Tensor)',
    'npu_prepare_wy_repr_bwd_da': '(Tensor k, Tensor v, Tensor beta, Tensor A, Tensor dw, Tensor du, Tensor g, *, int chunk_size, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> Tensor',
    'npu_prepare_wy_repr_bwd_full': '(Tensor k, Tensor v, Tensor beta, Tensor A, Tensor dA, Tensor dw, Tensor du, Tensor g, int chunk_size, *, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor, Tensor, Tensor)',
}

def register_missing_ops():
    lib = torch.library.Library('npu', 'FRAGMENT')
    registered = []
    for name, schema in SCHEMAS.items():
        if hasattr(torch.ops.npu, name):
            continue
        lib.define(name + schema)
        lib.impl(name, getattr(ascendc, name), 'PrivateUse1')
        registered.append(name)
    return lib, registered
