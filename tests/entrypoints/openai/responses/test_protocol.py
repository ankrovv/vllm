# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponseIncompleteEvent,
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
    serialize_messages,
)
from vllm.sampling_params import SamplingParams


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


def test_content_null_incomplete_response_contract() -> None:
    request = ResponsesRequest(input="hello")
    response = ResponsesResponse.from_request(
        request,
        SamplingParams(max_tokens=16),
        model_name="test-model",
        created_time=123,
        output=[],
        status="incomplete",
        stop_reason="content_null",
    )

    event = ResponseIncompleteEvent(
        type="response.incomplete",
        sequence_number=2,
        response=response,
    )
    dumped = event.model_dump(mode="json")

    assert dumped["response"]["status"] == "incomplete"
    assert dumped["response"]["stop_reason"] == "content_null"
    assert dumped["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
