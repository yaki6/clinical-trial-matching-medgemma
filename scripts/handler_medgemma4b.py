"""Custom HF Inference Endpoint handler for MedGemma 1.5 4B multimodal.

Fixes three server-side issues with the default image-text-to-text pipeline:
1. temperature leaks to AutoProcessor instead of model.generate()
2. Implicit float16 dtype causes cuBLAS misaligned-address CUDA crash on L4
3. Non-contiguous tensors trigger GEMM kernel failures

Deploy: push this file as handler.py to a HF model repo, then point the
inference endpoint to that repo. See scripts/deploy_medgemma4b_fix.py.
"""

from __future__ import annotations

import base64
import io
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class EndpointHandler:
    """Custom handler that routes generation params correctly."""

    def __init__(self, path: str = "") -> None:
        """Load model with explicit bfloat16 and fast image processor.

        If `path` contains model weights (endpoint downloaded them), load from
        there. Otherwise fall back to the gated Google repo via HF_TOKEN.
        """
        model_id = path or "google/medgemma-1.5-4b-it"
        token = os.environ.get("HF_TOKEN")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=token,
            use_fast=True,  # avoids edge-case preprocessing differences
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch.bfloat16,  # explicit — prevents float16 GEMM crash
            device_map="auto",
        )
        self.model.eval()

    def __call__(self, data: dict) -> list[dict]:
        """Handle inference request.

        Input formats supported:
          Text-only:  {"inputs": "<prompt>", "parameters": {"max_new_tokens": 512}}
          Multimodal: {"inputs": {"text": "<prompt>", "image": "<base64>"}, ...}
        """
        inputs = data.get("inputs", "")
        parameters = dict(data.get("parameters", {}))

        # ---- separate generation kwargs from processor kwargs ----
        temperature = parameters.pop("temperature", None)
        max_new_tokens = parameters.pop("max_new_tokens", 2048)
        do_sample = parameters.pop("do_sample", False)
        top_p = parameters.pop("top_p", None)
        top_k = parameters.pop("top_k", None)

        # ---- preprocess (processor only — no generation params) ----
        if isinstance(inputs, dict):
            text = inputs.get("text", "")
            image_b64 = inputs.get("image")
            if image_b64:
                image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
                processed = self.processor(text=text, images=image, return_tensors="pt")
            else:
                processed = self.processor(text=text, return_tensors="pt")
        else:
            processed = self.processor(text=str(inputs), return_tensors="pt")

        # Force contiguous layout — prevents cuBLAS misaligned-address crash
        processed = {
            k: v.contiguous().to(self.model.device)
            for k, v in processed.items()
            if isinstance(v, torch.Tensor)
        }

        # ---- build generation kwargs (model.generate only) ----
        gen_kwargs: dict = {"max_new_tokens": max_new_tokens}

        if temperature is not None and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
        elif do_sample:
            gen_kwargs["do_sample"] = True

        # ---- generate ----
        with torch.no_grad():
            output_ids = self.model.generate(**processed, **gen_kwargs)

        # Decode only new tokens (skip input prompt tokens)
        input_len = processed["input_ids"].shape[1]
        decoded = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        )

        return [{"generated_text": decoded}]
