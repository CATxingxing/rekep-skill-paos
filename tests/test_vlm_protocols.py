from __future__ import annotations

from vlm_client import _request_payload, _response_text, _responses_base


def test_prompt_defines_inside_region_as_zero_inside():
    from pathlib import Path

    prompt = Path(__file__).parents[1] / "nodes" / "rekep-planner" / "prompt_template.txt"
    text = prompt.read_text(encoding="utf-8")
    assert "returns exactly 0 for a point inside" in text
    assert 'relation "le", target 0' in text


def test_responses_endpoint_normalization():
    assert _responses_base("https://api.example") == "https://api.example/v1"
    assert _responses_base("https://api.example/v1") == "https://api.example/v1"
    assert _responses_base("https://api.example/v1/responses") == "https://api.example/v1"


def test_responses_multimodal_json_payload():
    endpoint, payload = _request_payload(
        {
            "protocol": "openai_responses",
            "api_base": "https://api.example",
            "model": "vision-model",
            "max_output_tokens": 4096,
            "reasoning_effort": "low",
        },
        "make a plan",
        "data:image/png;base64,AAAA",
    )
    assert endpoint == "https://api.example/v1/responses"
    assert payload["input"][0]["content"][0] == {"type": "input_text", "text": "make a plan"}
    assert payload["input"][0]["content"][1]["type"] == "input_image"
    assert payload["text"]["format"] == {"type": "json_object"}
    assert payload["reasoning"] == {"effort": "low"}


def test_responses_output_text_extraction():
    body = {
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": '{"stages":[]}'}],
        }]
    }
    assert _response_text(body, "openai_responses") == '{"stages":[]}'
