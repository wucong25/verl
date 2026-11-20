from pathlib import Path
p=Path('/vllm-ascend/vllm_ascend/ops/gdn.py')
s=p.read_text()
marker='    def _forward_core(\n'
method='''    def _forward_batch_invariant_gdn(self, q, k, v, g, beta, metadata, ssm_state):
        from vllm.config import get_current_vllm_config

        from vllm_ascend.ops.gdn_batch_invariant import GDNChunkReplay

        config = get_current_vllm_config()
        if not config.model_config.enforce_eager:
            raise RuntimeError("GDN chunk replay currently requires enforce_eager=True")
        if config.cache_config.enable_prefix_caching or get_pcp_group().world_size != 1:
            raise RuntimeError("GDN chunk replay requires prefix caching off and PCP=1")
        if metadata.spec_sequence_masks is not None:
            raise RuntimeError("GDN chunk replay does not yet support speculative decoding")
        boundaries = metadata.non_spec_query_start_loc.tolist()
        slots = metadata.non_spec_state_indices_tensor.reshape(-1).tolist()
        if len(slots) != len(boundaries) - 1:
            raise RuntimeError("Unsupported GDN cache-slot layout for chunk replay")
        pointer = ssm_state.data_ptr()
        if getattr(self, "_gdn_replay_storage", None) != pointer:
            self._gdn_replay_storage = pointer
            self._gdn_replay_sequences = {}
        flags = None if metadata.has_initial_state is None else metadata.has_initial_state.tolist()
        q, k = l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous())
        output = torch.empty_like(v)
        for seq, (slot, start, end) in enumerate(zip(slots, boundaries[:-1], boundaries[1:])):
            if start == end:
                continue
            is_new = flags is not None and not flags[seq]
            if is_new:
                self._gdn_replay_sequences[slot] = GDNChunkReplay(scale=k.shape[-1] ** -0.5)
            replay = self._gdn_replay_sequences.get(slot)
            if replay is None:
                raise RuntimeError("Missing GDN chunk history: unsupported cache restore or single-token initial prompt")
            out, final_state = replay.append(
                *(t[:, start:end].contiguous() for t in (q, k, v, g, beta))
            )
            output[:, start:end] = out
            ssm_state[slot].copy_(final_state[0].transpose(-1, -2))
        return output

'''
assert '_forward_batch_invariant_gdn' not in s
assert s.count(marker)==1
s=s.replace(marker,method+marker)
marker='            g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)\n'
assert s.count(marker)==1
new=marker+'''        if vllm_envs.VLLM_BATCH_INVARIANT:
            if split_conv:
                query_non_spec = torch.cat((decode_conv_qkv[0], query_non_spec), dim=1)
                key_non_spec = torch.cat((decode_conv_qkv[1], key_non_spec), dim=1)
                value_non_spec = torch.cat((decode_conv_qkv[2], value_non_spec), dim=1)
            output = self._forward_batch_invariant_gdn(
                query_non_spec, key_non_spec, value_non_spec, g, beta, attn_metadata, ssm_state
            )
            core_attn_out[:num_actual_tokens] = output.squeeze(0)
            maybe_save_kv_layer_to_connector("", [])
            return

'''
s=s.replace(marker,new)
import ast
ast.parse(s)
p.write_text(s)
print('GDN_REPLAY_PATCH_APPLIED')
