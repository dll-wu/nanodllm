"""Instantiated parameter inventories and architecture provenance; no model execution."""

import hashlib
import json

from . import ar_width_mup, nanollada_completep
from .model_parameterization import parameterization_fingerprint
from .model_profiles import NANOLLADA_PROFILE, nanollada_initialization_fingerprint

PARAMETER_COMPONENT_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("token_embedding", ("wte.",)),
    ("output_head", ("lm_head.",)),
    ("transformer_blocks", ("blocks.",)),
    ("final_norm", ("final_norm.",)),
    ("class_output_head", ("class_head.",)),
    ("length_head", ("length_head.",)),
    ("target_length_conditioning", ("target_length_embedding.", "target_distance_proj.")),
    ("mask_position_embedding", ("mask_position_embedding.",)),
    ("auxiliary_output_heads", ("intermediate_heads.",)),
    ("time_conditioning", ("noise_mlp.",)),
)
PARAMETER_COMPONENT_NAMES = tuple(component for component, _prefixes in PARAMETER_COMPONENT_PREFIXES) + (
    "other_parameters",
)
PARAMETER_ACCOUNTING_VIEW_COMPONENTS: dict[str, tuple[str, ...]] = {
    "total_parameters": PARAMETER_COMPONENT_NAMES,
    "transformer_body_parameters": tuple(
        component for component in PARAMETER_COMPONENT_NAMES if component not in {"token_embedding", "output_head"}
    ),
    "nanochat_scaling_parameters": ("transformer_blocks", "output_head"),
    "non_input_embedding_parameters": tuple(
        component for component in PARAMETER_COMPONENT_NAMES if component != "token_embedding"
    ),
}


class ModelAccounting:
    def parameter_inventory(self) -> dict[str, object]:
        """Inventory every instantiated named parameter exactly once.

        Components classify physical parameters. Accounting views are explicit
        sums of those components, so no architecture equation is an authority
        for counts or budgets.
        """
        components = {component: 0 for component in PARAMETER_COMPONENT_NAMES}
        component_members = {component: [] for component in PARAMETER_COMPONENT_NAMES}
        mtp = getattr(self, "mtp", None) is not None
        if mtp:
            components["mtp_parameters"] = 0
            component_members["mtp_parameters"] = []
        named_parameters: list[dict[str, object]] = []
        for name, parameter in self.named_parameters():
            component = "other_parameters"
            for candidate, prefixes in PARAMETER_COMPONENT_PREFIXES:
                if name.startswith(prefixes):
                    component = candidate
                    break
            if mtp and name.startswith("mtp."):
                component = "mtp_parameters"
            num_parameters = int(parameter.numel())
            components[component] += num_parameters
            component_members[component].append(name)
            named_parameters.append(
                {
                    "name": name,
                    "component": component,
                    "shape": list(parameter.shape),
                    "num_parameters": num_parameters,
                }
            )

        instantiated_total = sum(int(parameter.numel()) for parameter in self.parameters())
        component_total = sum(components.values())
        if component_total != instantiated_total:
            raise RuntimeError(
                "parameter inventory is incomplete: "
                f"components={component_total}, instantiated={instantiated_total}, "
                f"delta={instantiated_total - component_total}"
            )

        views: dict[str, dict[str, object]] = {}
        for view_name, view_components in PARAMETER_ACCOUNTING_VIEW_COMPONENTS.items():
            if mtp:
                view_components = (*view_components, "mtp_parameters")
            views[view_name] = {
                "components": list(view_components),
                "num_parameters": sum(components[component] for component in view_components),
            }
        if int(views["total_parameters"]["num_parameters"]) != instantiated_total:
            raise RuntimeError("total_parameters accounting view does not match the instantiated model")

        return {
            "method": "instantiated_named_parameters_numel",
            "components": components,
            "component_members": component_members,
            "views": views,
            "named_parameters": named_parameters,
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return component/view counts derived only from :meth:`parameter_inventory`.

        Historical flat keys remain because existing launchers and result
        readers consume them.
        """
        inventory = self.parameter_inventory()
        components = inventory["components"]
        views = inventory["views"]
        assert isinstance(components, dict)
        assert isinstance(views, dict)
        counts = {
            "token_embedding": int(components["token_embedding"]),
            "output_head": int(components["output_head"]),
            "transformer_blocks": int(components["transformer_blocks"]),
            "final_norm": int(components["final_norm"]),
            "class_output_head": int(components["class_output_head"]),
            "length_head": int(components["length_head"]),
            "target_length_conditioning": int(components["target_length_conditioning"]),
            "mask_position_embedding": int(components["mask_position_embedding"]),
            "auxiliary_output_heads": int(components["auxiliary_output_heads"]),
            "time_conditioning": int(components["time_conditioning"]),
            "other_parameters": int(components["other_parameters"]),
            "transformer_body_parameters": int(views["transformer_body_parameters"]["num_parameters"]),
            "nanochat_scaling_parameters": int(views["nanochat_scaling_parameters"]["num_parameters"]),
            "non_input_embedding_parameters": int(views["non_input_embedding_parameters"]["num_parameters"]),
            "total_parameters": int(views["total_parameters"]["num_parameters"]),
        }
        if "mtp_parameters" in components:
            counts["mtp_parameters"] = int(components["mtp_parameters"])
        return {
            **counts,
            # Compatibility aliases. Each is an alias of one named component or
            # accounting view above; none is computed independently.
            "wte": counts["token_embedding"],
            "lm_head": counts["output_head"],
            "class_head": counts["class_output_head"],
            "auxiliary_heads": counts["auxiliary_output_heads"],
            "transformer": counts["transformer_blocks"],
            "norm": counts["final_norm"],
            "noise_conditioning": counts["time_conditioning"],
            "non_embedding": counts["transformer_body_parameters"],
            "scaling": counts["nanochat_scaling_parameters"],
            "total": counts["total_parameters"],
        }

    def num_scaling_params(self) -> dict[str, int]:
        """Compatibility alias for older callers; prefer :meth:`parameter_counts`."""
        return self.parameter_counts()

    def estimate_flops(self, sequence_len: int | None = None, *, include_mtp: bool = True) -> int:
        counts = self.parameter_counts()
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head
        seq_len = self.config.sequence_len if sequence_len is None else int(sequence_len)
        if seq_len <= 0:
            raise ValueError("FLOP-estimate sequence length must be positive")
        if getattr(self, "nanochat_extras", None) is not None:
            from .nanochat_backbone import estimated_flops

            return estimated_flops(self, seq_len)
        n_layer = self.config.n_layer
        base = 6 * (counts["non_input_embedding_parameters"] - counts.get("mtp_parameters", 0))
        base += 12 * n_head * head_dim * seq_len * n_layer
        if self.recurrence.enabled:
            base += self.recurrence.extra_training_flops(
                [sum(p.numel() for p in block.parameters()) for block in self.blocks],
                n_embd=self.config.n_embd,
                sequence_len=seq_len,
            )
        return base + (self.mtp_flops(seq_len) if include_mtp else 0)

    def mtp_flops(self, sequence_len: int | None = None) -> int:
        """Dense training upper estimate, including the reused head's second matmul."""
        if getattr(self, "mtp", None) is None:
            return 0
        seq_len = self.config.sequence_len if sequence_len is None else int(sequence_len)
        if seq_len <= 0:
            raise ValueError("FLOP-estimate sequence length must be positive")
        parameters = sum(p.numel() for p in self.mtp.parameters())
        return 6 * (parameters + self.lm_head.weight.numel()) + 12 * self.config.n_embd * seq_len

    def architecture_fingerprint(self) -> dict[str, object]:
        """Return a stable architecture and accounting record for run provenance."""
        from .model import (
            embedding_head_init_std,
            residual_output_projection_init_std,
            resolved_attention_scale,
            resolved_attention_scale_rule,
            resolved_mlp_hidden_dim,
        )

        inventory = self.parameter_inventory()
        counts = self.parameter_counts()
        head_dim = int(self.config.n_embd) // int(self.config.n_head)
        payload: dict[str, object] = {
            "architecture": {
                "depth": int(self.config.n_layer),
                "n_embd": int(self.config.n_embd),
                "n_head": int(self.config.n_head),
                "head_dim": head_dim,
                "max_seq_len": int(self.config.sequence_len),
                "vocab_size": int(self.config.vocab_size),
                "padded_vocab_size": int(self.padded_vocab_size),
                "mlp_type": "squared_relu" if self.config.backbone_profile == NANOLLADA_PROFILE else "swiglu",
                "mlp_hidden_dim": resolved_mlp_hidden_dim(self.config),
                "causal": bool(self.config.causal),
                "embedding_head_tied": self.wte.weight is self.lm_head.weight,
                "output_factorization": str(self.config.output_factorization),
            },
            "attention": {
                "rope_base": float(self.rope_base),
                "qk_norm": "parameter_free_rms_norm_after_rope",
                "configured_scale": float(self.config.attention_scale),
                "resolved_scale": resolved_attention_scale(self.config),
                "scale_mode": resolved_attention_scale_rule(self.config),
                "local_attention_lower_layers": int(self.config.local_attention_lower_layers),
                "local_attention_window": int(self.config.local_attention_window),
            },
            "initialization": {
                "embedding_head_std": (
                    embedding_head_init_std(
                        int(self.config.n_embd), float(self.config.embedding_head_init_std_multiplier)
                    )
                    if self.config.parameterization_mode == "ordinary"
                    else float(self.config.parameterization_embedding_head_init_std)
                ),
                "block_matrix_std": (
                    (
                        float(self.config.parameterization_hidden_init_std)
                        if self.config.parameterization_mechanics_sha256
                        and float(self.config.parameterization_hidden_init_std) > 0
                        else residual_output_projection_init_std(int(self.config.n_layer))
                    )
                    if self.config.parameterization_mode == "ordinary"
                    else float(self.config.parameterization_hidden_init_std)
                ),
                "residual_output_projection_init": str(self.config.residual_output_projection_init),
            },
            "parameter_inventory": inventory,
            "parameter_counts": counts,
            "parameter_count_method": "instantiated_named_parameters_numel",
            "training_flops_per_token_estimate": self.estimate_flops(),
            "flops_estimate_method": "6x_non_input_embedding_parameter_view_plus_quadratic_attention",
        }
        if self.config.backbone_profile == NANOLLADA_PROFILE:
            payload["architecture"].update(
                {
                    "backbone_profile": NANOLLADA_PROFILE,
                    "input_norm": "parameter_free_rms_norm",
                    "final_norm": "parameter_free_rms_norm",
                }
            )
            payload["initialization"] = nanollada_initialization_fingerprint(self.config.n_embd)
            payload["initialization"]["output_head_std"] = nanollada_completep.head_std(self.config)
            if ar_width_mup.runtime_active(self.config):
                payload["initialization"]["embedding_parameter_dtype_rule"] = "float32_master_including_embedding"
                payload["architecture"]["norm_epsilon"] = ar_width_mup.NORM_EPSILON
                payload["architecture"]["norm_arithmetic"] = "fp32_reduce_cast_back_or_entirely_fp64_reference"
                payload["architecture"]["attention_backend"] = "sdpa_math_no_reduced_precision"
        if self.recurrence.enabled:
            payload["inner_recurrence"] = self.recurrence.fingerprint()
            payload["flops_estimate_method"] = "unique_parameters_plus_repeated_blocks_and_quadratic_attention"
        if getattr(self, "mtp", None) is not None:
            payload["mtp"] = {
                "mode": self.config.mtp_mode,
                "auxiliary_blocks": 1,
                "shared_input_embedding": True,
                "shared_output_head": True,
                "inference_branch": "skipped",
                "main_training_flops_per_token_estimate": self.estimate_flops(include_mtp=False),
                "auxiliary_training_flops_per_token_estimate": self.mtp_flops(),
            }
            payload["flops_estimate_method"] = "main_plus_mtp_dense_training_including_shared_head_reuse"
        if self.token_edit_adapters or self.token_edit_provenance_embedding is not None:
            token_edit_extensions: dict[str, object] = {
                "adapter_bottleneck_dim": int(self.config.token_edit_adapter_bottleneck_dim),
                "adapter_top_layers": int(self.config.token_edit_adapter_top_layers),
                "adapter_location": "after_each_of_the_configured_top_transformer_blocks"
                if self.token_edit_adapters
                else "disabled",
                "provenance_embedding": bool(self.token_edit_provenance_embedding is not None),
                "provenance_states": ["trusted_visible", "unresolved_mask", "model_draft"]
                if self.token_edit_provenance_embedding is not None
                else [],
            }
            if self.config.token_edit_provenance_confidence_bins > 0:
                token_edit_extensions["provenance_confidence_bins"] = int(
                    self.config.token_edit_provenance_confidence_bins
                )
            payload["token_edit_extensions"] = token_edit_extensions
        if self.config.bd3lm_self_conditioning_dim:
            payload["block_self_conditioning"] = {
                "bottleneck_dim": self.config.bd3lm_self_conditioning_dim,
                "method": "detached_same_state_soft_embedding_at_all_mask_inputs",
                "base_forward_passes_per_call_with_masks": 2,
                "base_flops_estimate_excludes_draft_pass": True,
            }
        resolved_parameterization = parameterization_fingerprint(self.config)
        if resolved_parameterization is not None:
            payload["parameterization_mechanics"] = resolved_parameterization
        if getattr(self, "nanochat_extras", None) is not None:
            from .nanochat_backbone import fingerprint

            fingerprint(self, payload)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {
            "schema_version": 2,
            "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
            **payload,
        }
