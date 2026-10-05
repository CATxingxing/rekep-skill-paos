#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_BASES = {
    "openai": "https://api.openai.com/v1",
    "openai_responses": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
    "moonshot": "https://api.moonshot.cn/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}
UNSUPPORTED = {"anthropic", "gemini", "azure_openai", "openai_codex", "github_copilot"}


def field(value: dict, snake: str, camel: str, default=None):
    return value.get(snake, value.get(camel, default))


def infer_provider(model: str) -> str:
    prefix = model.split("/", 1)[0].lower().replace("-", "_") if "/" in model else ""
    if prefix in DEFAULT_BASES or prefix in UNSUPPORTED or prefix in {"custom", "aihubmix", "minimax"}:
        return prefix
    lowered = model.lower()
    for name, keywords in {
        "openai": ("gpt", "openai"), "anthropic": ("claude", "anthropic"),
        "deepseek": ("deepseek",), "gemini": ("gemini",), "groq": ("groq",),
        "openrouter": ("openrouter",), "dashscope": ("qwen", "dashscope"),
    }.items():
        if any(keyword in lowered for keyword in keywords):
            return name
    return "custom"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bind ReKep planner to the current PAOS Agent VLM configuration.")
    parser.add_argument("--config", type=Path, default=Path.home() / ".PhyAgentOS/config.json")
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get("REKEP_RUNTIME_ROOT", Path.home() / ".paos-rekep")))
    arguments = parser.parse_args()
    value = json.loads(arguments.config.expanduser().read_text(encoding="utf-8"))
    defaults = value.get("agents", {}).get("defaults", {})
    model = field(defaults, "model", "model")
    provider = field(defaults, "provider", "provider", "auto")
    if not isinstance(model, str) or not model:
        raise SystemExit("PAOS agents.defaults.model is missing")
    if provider == "auto":
        provider = infer_provider(model)
    provider = str(provider).replace("-", "_")
    if provider in UNSUPPORTED:
        raise SystemExit(f"REKEP_VLM_UNSUPPORTED_PROVIDER: {provider} does not expose the required OpenAI-compatible multimodal chat-completions protocol")
    providers = value.get("providers", {})
    provider_config = providers.get(provider, providers.get("".join(part.title() if index else part for index, part in enumerate(provider.split("_"))), {}))
    if not isinstance(provider_config, dict):
        provider_config = {}
    api_key = field(provider_config, "api_key", "apiKey", "")
    api_base = field(provider_config, "api_base", "apiBase", None) or DEFAULT_BASES.get(provider)
    if provider == "custom" and not api_base:
        raise SystemExit("REKEP_VLM_UNSUPPORTED_PROVIDER: custom provider requires api_base")
    if not api_base:
        raise SystemExit(f"REKEP_VLM_UNSUPPORTED_PROVIDER: no OpenAI-compatible endpoint is known for {provider}")
    if provider not in {"ollama", "vllm"} and (not isinstance(api_key, str) or not api_key):
        raise SystemExit(f"REKEP_VLM_CONFIG_INVALID: {provider} requires an API key in the current PAOS configuration")
    protocol = "openai_responses" if provider == "openai_responses" else "openai_chat_completions"
    runtime = {
        "schema_version": 1,
        "provider": provider,
        "model": model,
        "protocol": protocol,
        "api_base": api_base,
        "api_key": api_key if isinstance(api_key, str) else "",
        "extra_headers": field(provider_config, "extra_headers", "extraHeaders", {}) or {},
        "timeout_s": 90,
        "max_output_tokens": int(field(defaults, "max_tokens", "maxTokens", 8192)),
        "reasoning_effort": field(defaults, "reasoning_effort", "reasoningEffort", None),
    }
    target = arguments.runtime_root.expanduser().resolve() / "run/rekep/vlm-provider.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(runtime, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, target)
    print(f"prepared ReKep runtime provider {provider}/{model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
