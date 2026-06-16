# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, importlib.util, platform

os.environ["UNSLOTH_IS_PRESENT"] = "1"

# ── Windows console UTF-8 safety ─────────────────────────────────────────────
# Legacy Windows consoles (cp1252) can't encode Unsloth's emoji/box-drawing
# glyphs and crash with UnicodeEncodeError. Force stdout/stderr to UTF-8 only on
# Windows and only when not already UTF-8; no-op elsewhere. errors="replace"
# guarantees we never crash on an unencodable glyph.
if platform.system() == "Windows":
    import sys as _sys
    for _name in ("stdout", "stderr"):
        _s = getattr(_sys, _name, None)
        try:
            _enc = (getattr(_s, "encoding", None) or "").lower()
            if _s is not None and hasattr(_s, "reconfigure") and "utf" not in _enc:
                _s.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass


def _is_mlx_available():
    # Transitional import barrier: keep non-Apple-Silicon imports from touching
    # unsloth_zoo until unsloth_zoo.mlx is import-safe on GPU hosts. Then this
    # can collapse back to the centralized zoo runtime call below.
    if (
        os.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") == "1"
        or platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or importlib.util.find_spec("mlx") is None
    ):
        return False
    try:
        from unsloth_zoo.mlx import is_mlx_available
    except ImportError:
        return False
    return is_mlx_available()


# Detect Apple Silicon + MLX before any torch/numpy imports
_IS_MLX = _is_mlx_available()

if _IS_MLX:
    try:
        import unsloth_zoo
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires `unsloth-zoo` with MLX modules. "
            "Reinstall with `pip install unsloth-zoo` or rerun install.sh."
        ) from _e
    # An older unsloth-zoo satisfies `import unsloth_zoo` but lacks the
    # mlx.trainer / mlx.loader submodules. Surface a friendly install hint
    # instead of a raw ImportError on the submodule path.
    try:
        from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
        from unsloth_zoo.mlx.loader import FastMLXModel
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires an unsloth-zoo build that includes "
            "`unsloth_zoo.mlx.trainer` and `unsloth_zoo.mlx.loader`. Upgrade with "
            "`pip install -U unsloth-zoo` or rerun install.sh."
        ) from _e

    # Load raw_text helpers without executing dataprep/__init__.py, which
    # imports synthetic.py -> torch and would defeat the torch-free MLX path.
    from pathlib import Path as _Path

    _raw_text_path = _Path(__file__).resolve().parent / "dataprep" / "raw_text.py"
    _raw_text_spec = importlib.util.spec_from_file_location("unsloth._mlx_raw_text", _raw_text_path)
    if _raw_text_spec is None or _raw_text_spec.loader is None:
        raise ImportError("Unsloth: could not load MLX raw_text dataprep helpers.")
    _raw_text = importlib.util.module_from_spec(_raw_text_spec)
    _raw_text_spec.loader.exec_module(_raw_text)
    RawTextDataLoader = _raw_text.RawTextDataLoader
    TextPreprocessor = _raw_text.TextPreprocessor
    del _raw_text, _raw_text_spec, _raw_text_path, _Path

    __version__ = unsloth_zoo.__version__
    DEVICE_TYPE = "mlx"

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FastMLXModel.from_pretrained(*args, **kwargs)

        @staticmethod
        def get_peft_model(*args, **kwargs):
            return FastMLXModel.get_peft_model(*args, **kwargs)

        @staticmethod
        def for_inference(*args, **kwargs):
            return args[0] if args else None

    class FastVisionModel(FastLanguageModel):
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("text_only", False)
            return FastMLXModel.from_pretrained(*args, **kwargs)

        @staticmethod
        def for_training(*args, **kwargs):
            return args[0] if args else None

    FastTextModel = FastLanguageModel
    FastModel = FastLanguageModel

    class FastSentenceTransformer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise NotImplementedError(
                "Unsloth: FastSentenceTransformer is not yet supported on MLX."
            )

        @staticmethod
        def get_peft_model(*args, **kwargs):
            raise NotImplementedError(
                "Unsloth: FastSentenceTransformer is not yet supported on MLX."
            )

    def is_bfloat16_supported():
        try:
            import mlx.core as mx
            name = mx.device_info().get("device_name", "") or ""
            return not name.startswith(("Apple M1", "Apple M2"))
        except Exception:
            return True

    is_bf16_supported = is_bfloat16_supported

    class UnslothVisionDataCollator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "Unsloth: UnslothVisionDataCollator is not used on MLX. "
                "Use the MLX trainer/data path instead."
            )

else:
    # GPU path: load everything from _gpu_init
    # ==========================================================
    # CUSTOM COMPATIBILITY PATCHES FOR TURING (RTX 2080 Ti)
    # ==========================================================
    import torch
    import os
    import sys

    # 1. Auto-disable Triton compile on Turing or older
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'

    # 2. Register TokenizersBackend alias in transformers
    try:
        import transformers
        from transformers import PreTrainedTokenizerFast
        if not hasattr(transformers, "TokenizersBackend"):
            transformers.TokenizersBackend = PreTrainedTokenizerFast
    except Exception:
        pass

    # 3. Patch SFTTrainer, SFTConfig, TrainingArguments, modeling utils, and peft_utils dynamically when imported
    class PatchingFinder(sys.meta_path.__class__):
        def find_spec(self, fullname, path, target=None):
            if fullname in ("trl.trainer.sft_config", "trl.trainer.sft_trainer", "transformers.training_args", "transformers.modeling_utils", "unsloth_zoo.peft_utils"):
                if self in sys.meta_path:
                    sys.meta_path.remove(self)
                try:
                    from importlib.util import find_spec
                    spec = find_spec(fullname)
                    if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
                        orig_exec = spec.loader.exec_module
                        def patched_exec(module):
                            orig_exec(module)
                            try:
                                if fullname == "trl.trainer.sft_config":
                                    _patch_sft_config(module)
                                elif fullname == "trl.trainer.sft_trainer":
                                    _patch_sft_trainer(module)
                                elif fullname == "transformers.training_args":
                                    _patch_training_args(module)
                                elif fullname == "transformers.modeling_utils":
                                    _patch_modeling_utils(module)
                                elif fullname == "unsloth_zoo.peft_utils":
                                    _patch_peft_utils(module)
                            except Exception as e:
                                print(f"[Unsloth Turing Patch] Error patching {fullname}: {e}")
                        spec.loader.exec_module = patched_exec
                        return spec
                finally:
                    if self not in sys.meta_path:
                        sys.meta_path.insert(0, self)
            return None

    def _patch_sft_config(module):
        if hasattr(module, "SFTConfig"):
            orig_init = module.SFTConfig.__init__
            def patched_init(self, *args, **kwargs):
                if 'max_seq_length' in kwargs:
                    val = kwargs.pop('max_seq_length')
                    if 'max_length' not in kwargs:
                        kwargs['max_length'] = val
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
                    if kwargs.get('bf16', False):
                        kwargs['bf16'] = False
                        kwargs['fp16'] = True
                orig_init(self, *args, **kwargs)
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
                    if getattr(self, 'bf16', False):
                        self.bf16 = False
                        self.fp16 = True
            module.SFTConfig.__init__ = patched_init

    def _patch_sft_trainer(module):
        if hasattr(module, "SFTTrainer"):
            from trl.trainer.utils import entropy_from_logits
            def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                mode = "train" if self.model.training else "eval"
                labels = inputs["labels"]
                inputs["use_cache"] = False
                from trl import SFTTrainer
                res = super(SFTTrainer, self).compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                )
                if isinstance(res, tuple):
                    loss, outputs = res
                else:
                    loss = res
                    outputs = None
                has_logits = hasattr(outputs, "logits") and type(outputs.logits).__name__ != "EmptyLogits"
                if not self.args.use_liger_kernel and has_logits:
                    with torch.no_grad():
                        per_token_entropy = entropy_from_logits(outputs.logits)
                        if "attention_mask" in inputs:
                            attention_mask = inputs["attention_mask"]
                            virtual_attention_mask = torch.ones(
                                attention_mask.size(0), self.num_virtual_tokens, device=attention_mask.device
                            )
                            attention_mask = torch.cat((virtual_attention_mask, attention_mask), dim=1)
                            entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                        elif "position_ids" in inputs:
                            entropy = torch.mean(per_token_entropy)
                        else:
                            raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
                    self._metrics[mode]["entropy"].append(entropy)
                if mode == "train":
                    if "attention_mask" in inputs:
                        num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
                    elif "position_ids" in inputs:
                        local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                        num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
                    else:
                        raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                    self._total_train_tokens += num_tokens_in_batch
                self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
                if not self.args.use_liger_kernel and has_logits:
                    with torch.no_grad():
                        if "shift_labels" in inputs:
                            shift_logits = outputs.logits.contiguous()
                            shift_labels = inputs["shift_labels"]
                        else:
                            shift_logits = outputs.logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                        shift_logits = shift_logits[:, self.num_virtual_tokens :, :]
                        predictions = shift_logits.argmax(dim=-1)
                        mask = shift_labels != -100
                        correct_predictions = (predictions == shift_labels) & mask
                        total_tokens = mask.sum()
                        correct_tokens = correct_predictions.sum()
                        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
                        total_sum = total_tokens.sum()
                        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                        self._metrics[mode]["mean_token_accuracy"].append(accuracy)
                        if self.aux_loss_enabled:
                            aux_loss = outputs.aux_loss
                            aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
                            self._metrics[mode]["aux_loss"].append(aux_loss)
                return (loss, outputs) if return_outputs else loss
            module.SFTTrainer.compute_loss = patched_compute_loss

    def _patch_training_args(module):
        if hasattr(module, "TrainingArguments"):
            orig_init = module.TrainingArguments.__init__
            def patched_init(self, *args, **kwargs):
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
                    if kwargs.get('bf16', False):
                        kwargs['bf16'] = False
                        kwargs['fp16'] = True
                orig_init(self, *args, **kwargs)
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
                    if getattr(self, 'bf16', False):
                        self.bf16 = False
                        self.fp16 = True
            module.TrainingArguments.__init__ = patched_init

    def _patch_modeling_utils(module):
        if hasattr(module, "PreTrainedModel"):
            orig_func = module.PreTrainedModel.from_pretrained.__func__
            @classmethod
            def patched_from_pretrained(cls, *args, **kwargs):
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
                    torch_dtype = kwargs.get('torch_dtype')
                    if torch_dtype in (torch.bfloat16, "bfloat16", "auto"):
                        kwargs['torch_dtype'] = torch.float16
                return orig_func(cls, *args, **kwargs)
            module.PreTrainedModel.from_pretrained = patched_from_pretrained

    def _patch_peft_utils(module):
        if hasattr(module, "get_peft_regex"):
            _orig_get_peft_regex = module.get_peft_regex
            def patched_get_peft_regex(
                model,
                finetune_vision_layers     : bool = True,
                finetune_language_layers   : bool = True,
                finetune_attention_modules : bool = True,
                finetune_mlp_modules       : bool = True,
                target_modules             = None,
                **kwargs
            ):
                if "attention_tags" in kwargs:
                    if "mixer" not in kwargs["attention_tags"]:
                        kwargs["attention_tags"] = list(kwargs["attention_tags"]) + ["mixer"]
                else:
                    kwargs["attention_tags"] = ["self_attn", "attention", "attn", "mixer"]

                if "mlp_tags" in kwargs:
                    if "mixer" not in kwargs["mlp_tags"]:
                        kwargs["mlp_tags"] = list(kwargs["mlp_tags"]) + ["mixer"]
                else:
                    kwargs["mlp_tags"] = ["mlp", "feed_forward", "ffn", "dense", "mixer"]

                return _orig_get_peft_regex(
                    model,
                    finetune_vision_layers     = finetune_vision_layers,
                    finetune_language_layers   = finetune_language_layers,
                    finetune_attention_modules = finetune_attention_modules,
                    finetune_mlp_modules       = finetune_mlp_modules,
                    target_modules             = target_modules,
                    **kwargs
                )
            module.get_peft_regex = patched_get_peft_regex

    sys.meta_path.insert(0, PatchingFinder())
    # ==========================================================

    from ._gpu_init import *
    from ._gpu_init import __version__
