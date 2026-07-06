# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for constrained-recipient normalization in the Harmony -> Responses
converter.

Regression coverage for vllm-project/vllm#45570: a constrained ``final`` message
(``<|channel|>final<|constrain|>json<|message|>{...}``) can be parsed by
openai-harmony with ``recipient == "<|constrain|>json"``. Without normalization
the converter routes it to :func:`_parse_mcp_call` and leaks the control token
into an ``mcp_call`` item's ``name``/``server_label``. It must instead be
returned as a normal output message.
"""

from openai_harmony import Message, Role

from vllm.entrypoints.openai.responses.harmony import (
    _normalize_recipient,
    harmony_to_response_output,
)


def test_normalize_recipient_strips_constrain_marker():
    # Bare marker -> no real recipient.
    assert _normalize_recipient("<|constrain|>json") is None
    # Marker after a real recipient -> keep the real part.
    assert (
        _normalize_recipient("functions.get_weather <|constrain|>json")
        == "functions.get_weather"
    )
    # Untainted recipients pass through unchanged.
    assert _normalize_recipient("functions.get_weather") == "functions.get_weather"
    assert _normalize_recipient("repo_browser.list") == "repo_browser.list"
    assert _normalize_recipient(None) is None
    assert _normalize_recipient("") == ""


def test_constrained_final_message_not_parsed_as_mcp_call():
    """A final message whose only 'recipient' is the leaked <|constrain|> marker
    must become a message, not an mcp_call (vllm-project/vllm#45570)."""
    payload = '{"name": "Science Fair", "date": "Friday"}'
    msg = (
        Message.from_role_and_content(Role.ASSISTANT, payload)
        .with_channel("final")
        .with_recipient("<|constrain|>json")
    )

    items = harmony_to_response_output(msg)

    assert len(items) == 1
    assert items[0].type == "message"
    assert items[0].content[0].text == payload
    assert all(getattr(item, "type", None) != "mcp_call" for item in items)
