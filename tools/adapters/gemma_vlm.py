"""Gemma 4 multimodal LLM caller (Gemma Terms of Use).

Thin wrapper around HuggingFace ``AutoProcessor`` / ``AutoModelForImageTextToText``
exposing a small ``generate(text, images)`` surface for one-shot multimodal
queries. Loads lazily on first call.

Defaults to ``google/gemma-4-31B-it`` with bitsandbytes NF4 4-bit quantization
(~18 GB on GPU, leaving headroom for KV cache + multiple image tiles on a
49 GB device). Pass ``quant="bf16"`` for full precision when memory allows.

When ``enable_thinking=True``, the chat template emits a reasoning channel
that this adapter strips and returns separately as ``(thinking, answer)``;
default ``max_new_tokens`` is raised (2048 vs 512) so the answer isn't
truncated by the thought prefix. Image inputs are placed BEFORE the text
block in the user content list — Gemma's processor expects this ordering
for stable grounding.

The first invocation downloads ~62 GB of bf16 weights (NF4 quantizes them
in-memory after download) from the chosen repo to the HuggingFace cache.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from PIL import Image as PILImage


_HF_REPO_DEFAULT = "google/gemma-4-31B-it"
_QUANT_CHOICES = ("nf4", "int8", "bf16", "fp16")

# Gemma 4's chat template renders thinking as ``<|channel>thought\n...\n<channel|>``
# blocks (asymmetric markers, channel-name follows the opener until ``\n``).
# A single completion may emit multiple blocks; thinking-text is interleaved
# with visible answer text.
_GEMMA_CHANNEL_OPEN = "<|channel>"
_GEMMA_CHANNEL_CLOSE = "<channel|>"

# Fallback symmetric marker pairs (older Gemma / gpt-oss style) used by
# ``_split_thinking_legacy`` when the Gemma 4 opener isn't present.
_LEGACY_MARKERS: tuple[tuple[str, str], ...] = (
    ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
)


class GemmaVLM:
    """Lazily-built Gemma 4 VLM for single-turn multimodal generation."""

    def __init__(
        self,
        model_id: str = _HF_REPO_DEFAULT,
        device: str = "cuda",
        quant: str = "nf4",
        cache_dir: str | None = None,
        cpu_offload: bool = False,
        attn_implementation: str | None = "sdpa",
    ):
        if quant not in _QUANT_CHOICES:
            raise ValueError(
                f"quant must be one of {_QUANT_CHOICES}, got {quant!r}"
            )
        self.model_id = model_id
        self.device = device
        self.quant = quant
        self.cache_dir = cache_dir
        self.cpu_offload = cpu_offload
        self.attn_implementation = attn_implementation
        self._model = None
        self._processor = None

    def _build(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(
            f"[gemma-vlm] loading {self.model_id} (quant={self.quant}, "
            f"cpu_offload={self.cpu_offload})"
        )

        kwargs: dict = {"cache_dir": self.cache_dir}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        if self.quant == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                # Skip vision tower + vision-to-LM projector — quantizing them
                # destroys image grounding (model hallucinates a blank image).
                # Match against full dotted module paths from named_modules().
                llm_int8_skip_modules=[
                    "model.vision_tower",
                    "model.embed_vision",
                    "lm_head",
                ],
            )
            kwargs["device_map"] = "auto"
        elif self.quant == "int8":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=[
                    "model.vision_tower",
                    "model.embed_vision",
                    "lm_head",
                ],
            )
            kwargs["device_map"] = "auto"
        elif self.quant == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.quant == "fp16":
            kwargs["torch_dtype"] = torch.float16

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **kwargs
        )

        # bnb places weights via device_map; only move for the dtype paths.
        if self.quant in ("bf16", "fp16"):
            if self.cpu_offload and hasattr(self._model, "enable_model_cpu_offload"):
                self._model.enable_model_cpu_offload()
            else:
                self._model = self._model.to(self.device)

        self._model.eval()

    def generate(
        self,
        text: str,
        images: list,
        *,
        system: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str | None, str]:
        """Run a single multimodal turn. Returns ``(thinking, answer)``.

        ``thinking`` is ``None`` when ``enable_thinking=False`` or when the
        template emitted no recognized thinking marker.
        """
        if self._model is None:
            self._build()
        assert self._model is not None and self._processor is not None

        pil_images = [_to_pil(img) for img in images]

        # Images BEFORE text — Gemma processor convention.
        user_content: list[dict] = [
            {"type": "image", "image": img} for img in pil_images
        ]
        user_content.append({"type": "text", "text": text})

        messages: list[dict] = []
        if system:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system}]}
            )
        messages.append({"role": "user", "content": user_content})

        if max_new_tokens is None:
            max_new_tokens = 2048 if enable_thinking else 512

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            # Some templates only accept enable_thinking under chat_template_kwargs.
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
        inputs = inputs.to(self._model.device)

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0, prompt_len:]
        decoded = self._processor.tokenizer.decode(
            new_tokens, skip_special_tokens=False
        )

        if enable_thinking:
            thinking, answer = _split_thinking(decoded)
        else:
            thinking, answer = None, decoded

        answer = _strip_specials(answer, self._processor.tokenizer)
        if thinking is not None:
            thinking = _strip_specials(thinking, self._processor.tokenizer)
        return thinking, answer

    def release(self) -> None:
        """Free the model + GPU cache. Safe to call when not built."""
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _to_pil(img) -> "PILImage.Image":
    from PIL import Image
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    raise TypeError(f"unsupported image input: {type(img).__name__}")


def _split_thinking(decoded: str) -> tuple[str | None, str]:
    """Split a decoded completion into ``(thinking, answer)``.

    Handles Gemma 4's asymmetric ``<|channel>name\\n...\\n<channel|>``
    blocks (potentially multiple per response). Falls back to legacy
    symmetric marker pairs (older Gemma / gpt-oss) and finally to
    ``(None, decoded)`` when nothing matches.
    """
    if _GEMMA_CHANNEL_OPEN in decoded:
        return _split_gemma4(decoded)
    for start, end in _LEGACY_MARKERS:
        i = decoded.find(start)
        if i == -1:
            continue
        j = decoded.find(end, i + len(start))
        if j == -1:
            return decoded[i + len(start):].strip(), ""
        thinking = decoded[i + len(start):j].strip()
        answer = (decoded[:i] + decoded[j + len(end):]).strip()
        return thinking, answer
    return None, decoded


def _split_gemma4(decoded: str) -> tuple[str | None, str]:
    """Parse Gemma 4's interleaved channel format into (thinking, answer).

    Format::

        <visible1><|channel>thought\\n<thinking1><channel|><visible2>...

    Channel name (``thought``, etc.) and trailing newline after the opener
    are stripped from the thinking text; same with the leading newline
    before the closer. Multiple thinking blocks are joined with newlines.
    An unclosed opener at end of text consumes the rest as thinking.
    """
    thinking_parts: list[str] = []
    visible_parts: list[str] = []
    rest = decoded
    while True:
        o = rest.find(_GEMMA_CHANNEL_OPEN)
        if o == -1:
            visible_parts.append(rest)
            break
        visible_parts.append(rest[:o])
        after_open = rest[o + len(_GEMMA_CHANNEL_OPEN):]
        # Drop the channel-name line ("thought\n", "final\n", etc.).
        nl = after_open.find("\n")
        if nl != -1:
            after_open = after_open[nl + 1:]
        c = after_open.find(_GEMMA_CHANNEL_CLOSE)
        if c == -1:
            # Unclosed — treat the rest as thinking, no further visible text.
            thinking_parts.append(after_open)
            break
        thinking_parts.append(after_open[:c])
        rest = after_open[c + len(_GEMMA_CHANNEL_CLOSE):]
    thinking = "\n".join(t.strip() for t in thinking_parts if t.strip())
    answer = "".join(visible_parts).strip()
    return (thinking or None, answer)


def _strip_specials(text: str, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


__all__ = ["GemmaVLM"]
