from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rekep_core.contracts import ContractError, read_json, runtime_root


class VLMError(ContractError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _configuration() -> dict[str, Any]:
    path = runtime_root() / "run" / "rekep" / "vlm-provider.json"
    value = read_json(path)
    if value.get("protocol") not in {"openai_chat_completions", "openai_responses"}:
        raise VLMError(
            "REKEP_VLM_UNSUPPORTED_PROVIDER",
            f"provider {value.get('provider')!r} is not configured for a supported multimodal OpenAI protocol",
        )
    for field in ("provider", "model", "api_base"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise VLMError("REKEP_VLM_CONFIG_INVALID", f"runtime VLM configuration requires {field}")
    if not isinstance(value.get("api_key"), str):
        raise VLMError("REKEP_VLM_CONFIG_INVALID", "runtime VLM configuration requires api_key")
    return value


def _responses_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/responses"):
        base = base[: -len("/responses")]
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _response_text(body: dict[str, Any], protocol: str) -> str | dict[str, Any]:
    if protocol == "openai_chat_completions":
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError("REKEP_VLM_INVALID_RESPONSE", "chat-completions response contains no assistant content") from exc
    chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    if not chunks and isinstance(body.get("output_text"), str):
        chunks.append(body["output_text"])
    if not chunks:
        raise VLMError("REKEP_VLM_INVALID_RESPONSE", "Responses API response contains no output_text")
    return "".join(chunks)


def _request_payload(config: dict[str, Any], prompt: str, image_url: str) -> tuple[str, dict[str, Any]]:
    if config["protocol"] == "openai_responses":
        payload: dict[str, Any] = {
            "model": config["model"],
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_url},
            ]}],
            "max_output_tokens": max(1, int(config.get("max_output_tokens", 8192))),
            "store": False,
            "text": {"format": {"type": "json_object"}},
        }
        reasoning_effort = config.get("reasoning_effort")
        if isinstance(reasoning_effort, str) and reasoning_effort not in {"", "none"}:
            payload["reasoning"] = {"effort": reasoning_effort}
        return _responses_base(config["api_base"]) + "/responses", payload
    return config["api_base"].rstrip("/") + "/chat/completions", {
        "model": config["model"],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
    }


def generate(instruction: str, snapshot: dict[str, Any], prompt_template: Path) -> tuple[list[dict[str, Any]], str, str]:
    config = _configuration()
    overlay = Path(snapshot["overlay_ref"])
    if not overlay.is_file():
        raise VLMError("REKEP_OBSERVATION_MISSING", f"bound observation overlay is missing: {overlay}")
    observation = json.dumps({
            "session_id": snapshot["session_id"],
            "scene_revision": snapshot["scene_revision"],
            "observation_id": snapshot["observation_id"],
            "frame_id": snapshot["frame_id"],
            "objects": snapshot["objects"],
            "keypoints": snapshot["keypoints"],
            "regions": snapshot["regions"],
        }, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        prompt_template.read_text(encoding="utf-8")
        .replace("{instruction}", instruction)
        .replace("{observation}", observation)
    )
    media = "image/png" if overlay.suffix.lower() == ".png" else "image/jpeg"
    image_url = f"data:{media};base64,{base64.b64encode(overlay.read_bytes()).decode('ascii')}"
    endpoint, payload = _request_payload(config, prompt, image_url)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config['api_key']}"}
    extra_headers = config.get("extra_headers") or {}
    if not isinstance(extra_headers, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in extra_headers.items()):
        raise VLMError("REKEP_VLM_CONFIG_INVALID", "extra_headers must be a string mapping")
    headers.update(extra_headers)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("timeout_s", 90))) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
        except OSError:
            detail = ""
        raise VLMError("REKEP_VLM_REQUEST_FAILED", f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise VLMError("REKEP_VLM_REQUEST_FAILED", str(exc)) from exc
    try:
        content = _response_text(body, config["protocol"])
        parsed = json.loads(content) if isinstance(content, str) else content
        stages = parsed["stages"]
    except VLMError:
        raise
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise VLMError("REKEP_VLM_INVALID_RESPONSE", "VLM response is not a constraint-program JSON object") from exc
    return stages, config["provider"], config["model"]
