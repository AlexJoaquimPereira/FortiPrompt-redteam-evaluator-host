"""
llm_adapter.py — Target LLM Adapter
=====================================
Abstracts over multiple LLM backends so the evaluator API can
call any of them through a single `LLMAdapter.generate()` coroutine.

Supported backends (set via LLM_BACKEND env var)
-------------------------------------------------
  openai        — OpenAI API (GPT-4o, GPT-3.5-turbo, etc.)
  azure_openai  — Azure OpenAI deployment
  anthropic     — Anthropic Claude API
  sagemaker     — AWS SageMaker real-time inference endpoint
  huggingface   — Local HuggingFace model via transformers (on-device)
  ollama        — Ollama local server (e.g. llama3, mistral)
  custom        — Any OpenAI-compatible endpoint (set LLM_BASE_URL)

Environment variables
---------------------
  LLM_BACKEND          One of the keys above (default: openai)
  LLM_MODEL            Model name / deployment name
  LLM_API_KEY          API key (OpenAI, Anthropic, Azure)
  LLM_BASE_URL         Custom base URL (for custom / Azure / Ollama)
  LLM_MAX_TOKENS       Max tokens in generated response (default: 512)
  LLM_TEMPERATURE      Sampling temperature (default: 0.7)
  # SageMaker specific
  SAGEMAKER_ENDPOINT   SageMaker endpoint name
  AWS_REGION           AWS region (default: us-east-1)
  # HuggingFace local
  HF_MODEL_ID          HuggingFace model id or local path
  HF_DEVICE            cuda / cpu / mps (default: auto)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMAdapterConfig:
    backend:          str   = "openai"
    model:            str   = "gpt-4o-mini"
    api_key:          str   = ""
    base_url:         Optional[str] = None
    max_tokens:       int   = 512
    temperature:      float = 0.7
    # SageMaker
    sagemaker_endpoint: str = ""
    aws_region:         str = "us-east-1"
    # HuggingFace local
    hf_model_id:  str = ""
    hf_device:    str = "auto"

    @classmethod
    def from_env(cls) -> "LLMAdapterConfig":
        return cls(
            backend           = os.getenv("LLM_BACKEND",          "openai"),
            model             = os.getenv("LLM_MODEL",            "gpt-4o-mini"),
            api_key           = os.getenv("LLM_API_KEY",          ""),
            base_url          = os.getenv("LLM_BASE_URL",         None),
            max_tokens        = int(os.getenv("LLM_MAX_TOKENS",   "512")),
            temperature       = float(os.getenv("LLM_TEMPERATURE","0.7")),
            sagemaker_endpoint= os.getenv("SAGEMAKER_ENDPOINT",   ""),
            aws_region        = os.getenv("AWS_REGION",           "us-east-1"),
            hf_model_id       = os.getenv("HF_MODEL_ID",          ""),
            hf_device         = os.getenv("HF_DEVICE",            "auto"),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class LLMAdapter:
    """
    Thin async wrapper over multiple LLM backends.

    Usage
    -----
        adapter = LLMAdapter(LLMAdapterConfig.from_env())
        response_text = await adapter.generate("Tell me how to do X.")
    """

    def __init__(self, config: LLMAdapterConfig):
        self._cfg = config
        self._hf_pipeline = None   # lazy-loaded HuggingFace pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._cfg.model

    async def generate(
        self,
        prompt:         str,
        system_prompt:  Optional[str]            = None,
        model_override: Optional[str]            = None,
        messages:       Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a completion from the configured backend.

        Parameters
        ----------
        prompt         : The user prompt.
        system_prompt  : Optional system message override.
        model_override : Override the config model for this single call.
        messages       : Full prior message history (OpenAI format).
                         If supplied, `prompt` is appended as a new user turn.
        """
        backend = self._cfg.backend.lower()
        dispatch = {
            "openai":       self._openai,
            "azure_openai": self._openai,      # same client, different base_url
            "custom":       self._openai,      # OpenAI-compatible
            "anthropic":    self._anthropic,
            "sagemaker":    self._sagemaker,
            "huggingface":  self._huggingface,
            "ollama":       self._openai,      # Ollama speaks OpenAI protocol
        }
        fn = dispatch.get(backend)
        if fn is None:
            raise ValueError(f"Unknown LLM_BACKEND: {backend!r}. "
                             f"Choose from: {', '.join(dispatch)}")
        return await fn(
            prompt=prompt,
            system_prompt=system_prompt,
            model_override=model_override,
            messages=messages,
        )

    # ------------------------------------------------------------------
    # OpenAI / Azure / Ollama / Custom
    # ------------------------------------------------------------------

    async def _openai(
        self, prompt: str, system_prompt: Optional[str],
        model_override: Optional[str], messages: Optional[list[dict]]
    ) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        cfg = self._cfg
        kwargs: dict[str, Any] = {}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        # Ollama default base URL
        if cfg.backend == "ollama" and not cfg.base_url:
            kwargs["base_url"] = "http://localhost:11434/v1"
            kwargs["api_key"] = kwargs.get("api_key") or "ollama"

        client = AsyncOpenAI(**kwargs)
        model  = model_override or cfg.model

        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if messages:
            # Prepend existing history (excluding any prior system msgs if we added one)
            history = [m for m in messages if m.get("role") != "system" or not system_prompt]
            msgs += history
        msgs.append({"role": "user", "content": prompt})

        completion = await client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        return completion.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    async def _anthropic(
        self, prompt: str, system_prompt: Optional[str],
        model_override: Optional[str], messages: Optional[list[dict]]
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        cfg    = self._cfg
        client = anthropic.AsyncAnthropic(api_key=cfg.api_key or None)
        model  = model_override or cfg.model

        # Anthropic expects alternating user/assistant turns
        msgs: list[dict] = []
        if messages:
            msgs = [{"role": m["role"], "content": m.get("content", "")}
                    for m in messages if m.get("role") in ("user", "assistant")]
        msgs.append({"role": "user", "content": prompt})

        response = await client.messages.create(
            model=model,
            max_tokens=cfg.max_tokens,
            system=system_prompt or anthropic.NOT_GIVEN,
            messages=msgs,
        )
        return response.content[0].text if response.content else ""

    # ------------------------------------------------------------------
    # AWS SageMaker
    # ------------------------------------------------------------------

    async def _sagemaker(
        self, prompt: str, system_prompt: Optional[str],
        model_override: Optional[str], messages: Optional[list[dict]]
    ) -> str:
        import asyncio
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("pip install boto3") from exc

        cfg      = self._cfg
        endpoint = cfg.sagemaker_endpoint
        if not endpoint:
            raise ValueError("SAGEMAKER_ENDPOINT env var not set.")

        # Build payload in OpenAI messages format — most SageMaker JumpStart
        # models and TGI containers accept this natively.
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if messages:
            msgs += [m for m in messages if m.get("role") in ("user", "assistant")]
        msgs.append({"role": "user", "content": prompt})

        body = json.dumps({
            "messages":    msgs,
            "max_tokens":  cfg.max_tokens,
            "temperature": cfg.temperature,
        }).encode("utf-8")

        def _invoke():
            sm = boto3.client("sagemaker-runtime", region_name=cfg.aws_region)
            return sm.invoke_endpoint(
                EndpointName=endpoint,
                ContentType="application/json",
                Body=body,
            )

        loop     = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _invoke)
        result   = json.loads(response["Body"].read())

        # Handle both {choices:[{message:{content:...}}]} and {generated_text:...}
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return result.get("generated_text", str(result))

    # ------------------------------------------------------------------
    # Local HuggingFace (on-device)
    # ------------------------------------------------------------------

    async def _huggingface(
        self, prompt: str, system_prompt: Optional[str],
        model_override: Optional[str], messages: Optional[list[dict]]
    ) -> str:
        import asyncio
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError("pip install transformers") from exc

        cfg      = self._cfg
        model_id = model_override or cfg.hf_model_id or cfg.model

        if self._hf_pipeline is None:
            logger.info("[HF] Loading %s (device=%s) …", model_id, cfg.hf_device)
            self._hf_pipeline = hf_pipeline(
                "text-generation",
                model=model_id,
                device_map=cfg.hf_device,
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                do_sample=cfg.temperature > 0,
            )

        # Build a chat string from the message history
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        if messages:
            for m in messages:
                role    = m.get("role", "user").capitalize()
                content = m.get("content", "")
                full_prompt += f"{role}: {content}\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        def _run():
            return self._hf_pipeline(full_prompt)[0]["generated_text"]

        loop   = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, _run)
        # Strip the prompt from the output
        if output.startswith(full_prompt):
            output = output[len(full_prompt):]
        return output.strip()
