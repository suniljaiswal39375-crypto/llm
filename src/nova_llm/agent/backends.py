"""Pluggable LLM chat backends for Nova-Search Agent & FastAPI server.

Supports:
  1. `NovaLMBackend`: Native PyTorch `NovaLM` + `ByteBPETokenizer` (including imported HF weights).
  2. `TransformersBackend`: Real Hugging Face / Unsloth checkpoints (e.g. Llama 3.2 3B Instruct 4-bit
     + LoRA adapters from Google Drive) with native chat template & streaming.
"""

from __future__ import annotations

from threading import Thread
from typing import Any, Iterator, Protocol

import torch

from nova_llm.inference.generator import generate, stream_generate
from nova_llm.inference.sampler import SamplingParams
from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer


class ChatGeneratorBackend(Protocol):
    """Uniform interface allowing SearchAgent & FastAPI to run on NovaLM OR Llama-3.2-3B/HF models."""

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> str: ...

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]: ...

    def info(self) -> dict[str, Any]: ...


class NovaLMBackend:
    """Native PyTorch NovaLM backend."""

    def __init__(self, model: NovaLM, tokenizer: ByteBPETokenizer, model_name: str = "nova-search") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model.eval()

    def _stop_ids(self) -> list[int]:
        ids = [self.tokenizer.eos_id]
        im_end = self.tokenizer.special_to_id.get("<|im_end|>")
        if im_end is not None:
            ids.append(im_end)
        return ids

    def _encode_messages(self, messages: list[dict[str, str]]) -> torch.Tensor:
        prompt_text = self.tokenizer.format_chat(messages, add_generation_prompt=True)
        toks = self.tokenizer.encode(prompt_text, add_bos=True, allow_special=True)
        max_ctx = self.model.cfg.max_seq_len - 16
        if len(toks) > max_ctx:
            toks = toks[-max_ctx:]
        return torch.tensor([toks], dtype=torch.long)

    def _clean(self, text: str) -> str:
        return text.replace("<|im_end|>", "").replace("<|eos|>", "")

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> str:
        ids = self._encode_messages(messages)
        params = SamplingParams(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self._stop_ids(),
            stop_strings=["<|im_end|>", "<|eos|>"],
        )
        res = generate(self.model, ids, params=params, tokenizer=self.tokenizer, use_cache=True)
        return self._clean(res.text or "").strip()

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        ids = self._encode_messages(messages)
        params = SamplingParams(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self._stop_ids(),
            stop_strings=["<|im_end|>", "<|eos|>"],
        )
        for chunk in stream_generate(self.model, ids, self.tokenizer, params=params):
            clean = self._clean(chunk)
            if clean:
                yield clean

    def info(self) -> dict[str, Any]:
        return {
            "backend": "nova_lm",
            "model": self.model_name,
            "params": self.model.count_parameters(),
            "vocab_size": self.tokenizer.vocab_size,
        }


class TransformersBackend:
    """Hugging Face / Unsloth backend (e.g. Llama 3.2 3B Instruct 4-bit + Google Drive LoRA).

    Accepts either an already-loaded `(model, tokenizer)` pair (ideal inside Google Colab with
    Unsloth `FastLanguageModel`) or loads directly from `model_name_or_path` (+ optional `lora_dir`).
    """

    def __init__(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        model_name_or_path: str | None = None,
        lora_dir: str | None = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        model_name: str | None = None,
    ) -> None:
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.model_name = model_name or getattr(getattr(model, "config", None), "_name_or_path", "llama-3.2-3b")
        else:
            if not model_name_or_path:
                raise ValueError("Provide either (model, tokenizer) or model_name_or_path")
            self.model_name = model_name or model_name_or_path
            self.model, self.tokenizer = self._load_hf(
                model_name_or_path, lora_dir=lora_dir, load_in_4bit=load_in_4bit, device_map=device_map
            )

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

    @staticmethod
    def _load_hf(
        model_name_or_path: str,
        lora_dir: str | None,
        load_in_4bit: bool,
        device_map: str,
    ) -> tuple[Any, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformersBackend requires `transformers` (and optionally `peft`/`bitsandbytes`). "
                "Install with: pip install transformers accelerate peft bitsandbytes"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(lora_dir or model_name_or_path)
        kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if lora_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_dir)
        model.eval()
        return model, tokenizer

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        # Normalize 'tool' role cleanly for Llama-3.1/3.2 templates that expect user/assistant/system
        norm_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "tool":
                norm_msgs.append({"role": "user", "content": f"Search Results:\n{content}\nNow answer the user's question concisely with inline citations like [1] and [2]."})
            else:
                norm_msgs.append({"role": role, "content": content})

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            return str(self.tokenizer.apply_chat_template(norm_msgs, tokenize=False, add_generation_prompt=True))

        # Clean fallback prompt format if tokenizer has no chat_template
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in norm_msgs]
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def _eos_token_ids(self) -> list[int] | int | None:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        ids: list[int] = []
        if isinstance(eos, int):
            ids.append(eos)
        elif isinstance(eos, list):
            ids.extend(eos)
        # Include Llama-3 <|eot_id|> if present
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            for tok in ("<|eot_id|>", "<|im_end|>"):
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0 and tid not in ids:
                    ids.append(tid)
        return ids if len(ids) > 1 else (ids[0] if ids else None)

    def _device(self) -> torch.device:
        dev = getattr(self.model, "device", None)
        if dev is not None:
            return dev
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ) -> str:
        prompt = self._format_messages(messages)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        device = self._device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._eos_token_ids(),
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0.0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out_ids[0, prompt_len:]
        return str(self.tokenizer.decode(new_ids, skip_special_tokens=True)).strip()

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ) -> Iterator[str]:
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            yield self.generate_chat(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            return

        prompt = self._format_messages(messages)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        device = self._device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._eos_token_ids(),
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0.0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            gen_kwargs["do_sample"] = False

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join(timeout=5.0)

    def info(self) -> dict[str, Any]:
        total = 0
        try:
            total = sum(p.numel() for p in self.model.parameters())
        except Exception:
            pass
        return {
            "backend": "transformers",
            "model": self.model_name,
            "params": {"total": total},
            "vocab_size": int(getattr(self.tokenizer, "vocab_size", 0)),
        }
