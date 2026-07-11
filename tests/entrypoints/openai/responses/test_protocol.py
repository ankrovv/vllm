# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)


@pytest.mark.parametrize(
    ("part_type", "image_url", "expected_detail"),
    [
        ("input_image", "data:image/png;base64,AAAA", "auto"),
        ("input_image", {"url": "data:image/png;base64,AAAA"}, "auto"),
        (
            "image_url",
            {"url": "data:image/png;base64,AAAA", "detail": "low"},
            "low",
        ),
        (
            "image_url",
            {"url": "data:image/png;base64,AAAA", "detail": "high"},
            "high",
        ),
    ],
)
def test_input_image_defaults_detail_and_accepts_chat_format(
    part_type: str, image_url: str | dict[str, str], expected_detail: str
) -> None:
    # Regression test for #46631: flat Responses images with missing detail and
    # chat-completions image parts must both be accepted.
    req = ResponsesRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {
                            "type": part_type,
                            "image_url": image_url,
                        },
                    ],
                }
            ],
        }
    )
    image_part = req.input[0]["content"][1]
    # type normalized to input_image; nested {"url": X} flattened to X;
    # required `detail` defaulted to "auto".
    assert image_part["type"] == "input_image"
    assert image_part["image_url"] == "data:image/png;base64,AAAA"
    assert image_part["detail"] == expected_detail


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]
