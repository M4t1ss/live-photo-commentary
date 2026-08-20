import importlib.util
import logging

import torch
from transformers import AutoModel

from ..local_describer import LocalDescriber

log = logging.getLogger(__name__)


class InternVLLocalDescriber(LocalDescriber):
    """Adapter for InternVL3.5 (and similar InternVLChat-architecture) models.

    These models use trust_remote_code with a custom InternVLChatConfig that is
    not registered in HF's AutoModelForImageTextToText/pipeline mapping, so the
    pipeline describer cannot load them.  Instead we use AutoModel directly and
    call the model's own .chat() method with pre-tiled pixel_values tensors.
    """

    uses_processor = False
    tokenizer_extra_kwargs = {"use_fast": False}

    def _create_model(self, quantization_config, attn_implementation, device):
        from transformers import PreTrainedModel

        model_kwargs = {
            "device_map": "cuda" if device == "cuda" and not self.device_param else None,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            # InternVL uses its own use_flash_attn kwarg, not the HF standard one
            "use_flash_attn": importlib.util.find_spec("flash_attn") is not None,
        } | self.model_kwargs
        log.info("model_kwargs: %s", model_kwargs)

        # InternVL's remote code was written for transformers 4.x, which called
        # post_init() automatically from PreTrainedModel.__init__.  Transformers
        # 5.x dropped that auto-call; each model must call self.post_init()
        # explicitly.  InternVL never does, so all_tied_weights_keys (set inside
        # post_init) is never initialised, crashing multiple spots in from_pretrained
        # across all device/quantisation paths.
        # Fix: resolve the actual InternVLChatModel class and wrap its __init__
        # to call PreTrainedModel.post_init() when it finishes — restoring the
        # transformers 4.x behaviour for this model only.
        _patched_cls = None
        _orig_cls_init = None
        try:
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            config = AutoConfig.from_pretrained(self.load_path, trust_remote_code=True)
            model_class_ref = getattr(config, "auto_map", {}).get("AutoModel")
            if model_class_ref:
                model_cls = get_class_from_dynamic_module(model_class_ref, self.load_path)
                _orig_cls_init = model_cls.__init__

                def _compat_init(self_m, cfg, *args, **kwargs):
                    _orig_cls_init(self_m, cfg, *args, **kwargs)
                    if not hasattr(self_m, "all_tied_weights_keys"):
                        PreTrainedModel.post_init(self_m)

                model_cls.__init__ = _compat_init
                _patched_cls = model_cls
        except Exception as e:
            log.warning("InternVL post_init compat patch failed: %s", e)

        try:
            return AutoModel.from_pretrained(self.load_path, **model_kwargs).eval()
        finally:
            if _patched_cls is not None:
                _patched_cls.__init__ = _orig_cls_init

    def _preprocess_image(self, image, max_num=12, input_size=448):
        """Dynamic-tiling image preprocessing used by InternVL.

        Splits the image into at most max_num tiles that best match its aspect
        ratio, plus an overall thumbnail when more than one tile is produced.
        Returns a [num_tiles, 3, input_size, input_size] tensor (not yet on device).
        """
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        image = image.convert("RGB")
        w, h = image.size
        aspect = w / h

        target_ratios = sorted(
            {(i, j) for n in range(1, max_num + 1)
             for i in range(1, n + 1) for j in range(1, n + 1)
             if 1 <= i * j <= max_num},
            key=lambda r: r[0] * r[1],
        )
        # prefer fewer tiles on tie (min area), break ties by closer aspect ratio
        best = min(target_ratios, key=lambda r: (abs(aspect - r[0] / r[1]), -r[0] * r[1]))

        resized = image.resize((input_size * best[0], input_size * best[1]))
        tiles = [
            transform(resized.crop((
                c * input_size, r * input_size,
                (c + 1) * input_size, (r + 1) * input_size,
            )))
            for r in range(best[1]) for c in range(best[0])
        ]
        if len(tiles) > 1:
            tiles.append(transform(image.resize((input_size, input_size))))

        return torch.stack(tiles)

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        images = images or []

        self.model.system_message = system_prompt or ""

        if images:
            tile_batches = [self._preprocess_image(img) for img in images]
            pixel_values = torch.cat(tile_batches).to(self.device, dtype=self.dtype)
            chat_kwargs = {} if len(images) == 1 else {"num_patches_list": [len(t) for t in tile_batches]}
        else:
            pixel_values = None
            chat_kwargs = {}

        return self.model.chat(
            self.tokenizer, pixel_values, user_prompt,
            generation_config=self.generation_args,
            **chat_kwargs,
        )
