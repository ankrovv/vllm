# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Responses API tool-calling request adjustment.

Covers two bugs on the ``/v1/responses`` path that broke streaming tool
calling for parsers relying on special-token delimiters (Gemma4):

1. :class:`Gemma4ToolParser.adjust_request` used an
   ``isinstance(request, ChatCompletionRequest)`` guard, so a
   :class:`ResponsesRequest` with tools never had
   ``skip_special_tokens`` flipped to ``False``. The default (``True``)
   stripped ``<|tool_call>`` / ``<tool_call|>`` delimiters, causing
   :meth:`Gemma4ToolParser.extract_tool_calls_streaming` to fall through
   to the content branch and leak the raw ``call:fn{...}`` body via
   ``response.output_text.delta``.

2. :meth:`ToolParser.adjust_request` built
   :class:`ResponseTextConfig` in two steps (bare constructor then
   ``.format = ...``). Under Pydantic v2 the later assignment is not
   tracked in ``__fields_set__``, which can drop the nested config from
   ``model_dump``. It also passed a ``description`` kwarg carrying the
   wrong-purpose string ``"Response format for tool calling"``.

3. :class:`Gemma4EngineToolParser` must enforce ``required`` and named tool
   choice with the standard JSON constraint. Automatic tool choice continues
   to use Gemma's native ``<|tool_call>`` syntax.
"""

from __future__ import annotations

from typing import Any

from openai.types.responses.tool_param import FunctionToolParam

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser, StreamState
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.gemma4_engine_tool_parser import (
    Gemma4EngineToolParser as Gemma4ToolParser,
)


def _get_weather_tool() -> FunctionToolParam:
    return FunctionToolParam(
        type="function",
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        strict=True,
    )


def _build_responses_request(*, tool_choice: str | dict[str, Any]) -> ResponsesRequest:
    return ResponsesRequest(
        model="gemma4-test",
        input=[{"role": "user", "content": "What is the weather in Hanoi?"}],
        tools=[_get_weather_tool()],
        tool_choice=tool_choice,
        stream=True,
        max_output_tokens=200,
    )


def _build_chat_request(
    *,
    tool_choice: str | dict[str, Any],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> ChatCompletionRequest:
    data: dict[str, Any] = {
        "model": "gemma4-test",
        "messages": [{"role": "user", "content": "What is the weather in Hanoi?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": tool_choice,
    }
    if chat_template_kwargs is not None:
        data["chat_template_kwargs"] = chat_template_kwargs
    return ChatCompletionRequest.model_validate(data)


class _StubTokenizer:
    """Minimal tokenizer stub to satisfy ``Gemma4EngineToolParser.__init__``."""

    def get_vocab(self) -> dict[str, int]:
        return {
            "<|tool_call>": 256_000,
            "<tool_call|>": 256_001,
            '<|"|>': 52,
            "<|channel>": 256_002,
            "<channel|>": 256_003,
        }


class _Gemma4DelegatingParser(DelegatingParser):
    tool_parser_cls = Gemma4ToolParser


def test_gemma4_adjust_request_sets_skip_special_tokens_on_responses() -> None:
    """``Gemma4ToolParser.adjust_request`` must flip
    ``skip_special_tokens=False`` for both ``ChatCompletionRequest`` and
    ``ResponsesRequest`` so that ``<|tool_call>`` delimiters reach the
    streaming extractor. The previous
    ``isinstance(ChatCompletionRequest)`` guard omitted the Responses
    path, causing raw ``call:fn{...}`` text to leak via
    ``response.output_text.delta``.
    """
    parser = Gemma4ToolParser(_StubTokenizer())

    request = _build_responses_request(tool_choice="auto")
    assert request.skip_special_tokens is True, (
        "Precondition: ResponsesRequest.skip_special_tokens default is True"
    )

    parser.adjust_request(request)

    assert request.skip_special_tokens is False


def test_tool_parser_adjust_request_builds_valid_response_text_config() -> None:
    """``ToolParser.adjust_request`` must produce a ``ResponseTextConfig``
    whose dumped form contains the JSON schema under the ``schema`` alias
    and does not leak the unrelated ``"Response format for tool calling"``
    description string that the previous two-step construction injected.
    """
    parser = ToolParser.__new__(ToolParser)
    parser.model_tokenizer = None

    request = _build_responses_request(tool_choice="required")
    ToolParser.adjust_request(parser, request)

    assert request.text is not None
    assert request.text.format is not None
    assert request.text.format.type == "json_schema"

    dump: dict[str, Any] = request.text.model_dump(mode="json", by_alias=True)
    fmt = dump.get("format") or {}
    assert fmt.get("type") == "json_schema"
    assert fmt.get("name") == "tool_calling_response"
    assert fmt.get("strict") is True
    # Nested config must be present under the alias. Two-step Pydantic v2
    # construction could drop it from __fields_set__.
    assert "schema" in fmt and isinstance(fmt["schema"], dict)
    # The old code passed a wrong-purpose string; valid field should now
    # either be absent or None (the openai-python default).
    assert fmt.get("description") in (None, "")


def test_gemma4_required_enforces_structured_outputs_chatcompletion() -> None:
    """required + ChatCompletion must constrain output to a tool-call list."""
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(tool_choice="required")

    parser.adjust_request(request)

    assert parser.supports_required_and_named is True
    assert parser.engine_based_streaming is True
    assert parser.cumulative_tool_streaming_for_required_and_named is True
    assert request.structured_outputs is not None
    assert request.structured_outputs.json is not None
    assert request.skip_special_tokens is False


def test_gemma4_named_enforces_structured_outputs_chatcompletion() -> None:
    """named + ChatCompletion must constrain output to function arguments."""
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )

    parser.adjust_request(request)

    assert request.structured_outputs is not None
    assert request.structured_outputs.json is not None
    assert request.skip_special_tokens is False


def test_gemma4_required_enforces_structured_outputs_responses() -> None:
    """required + Responses must constrain output to a tool-call list."""
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_responses_request(tool_choice="required")

    parser.adjust_request(request)

    assert request.text is not None
    assert request.text.format is not None
    assert request.text.format.type == "json_schema"
    assert request.skip_special_tokens is False


def test_gemma4_named_enforces_structured_outputs_responses() -> None:
    """named + Responses must constrain output to function arguments."""
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_responses_request(
        tool_choice={"type": "function", "name": "get_weather"}
    )

    parser.adjust_request(request)

    assert request.text is not None
    assert request.text.format is not None
    assert request.text.format.type == "json_schema"
    assert request.skip_special_tokens is False


def test_gemma4_required_parses_constrained_output() -> None:
    """The constrained required output must become a function call."""
    request = _build_responses_request(tool_choice="required")
    parser = _Gemma4DelegatingParser(_StubTokenizer(), tools=request.tools)

    reasoning, content, tool_calls = parser.parse(
        '[{"name":"get_weather","parameters":{"city":"Paris"}}]',
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == '{"city": "Paris"}'


def test_gemma4_named_parses_constrained_output() -> None:
    """The constrained named output must use the requested function name."""
    request = _build_chat_request(
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    parser = _Gemma4DelegatingParser(_StubTokenizer(), tools=request.tools)

    reasoning, content, tool_calls = parser.parse(
        '{"city":"Paris"}', request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == '{"city":"Paris"}'


def test_gemma4_accumulates_only_for_forced_tool_streaming() -> None:
    """Gemma keeps native delta streaming for auto and accumulates forced JSON."""
    parser = _Gemma4DelegatingParser(_StubTokenizer())
    parser._stream_state.reasoning_ended = True

    assert parser._uses_cumulative_tool_stream_state(
        _build_responses_request(tool_choice="required")
    )
    assert parser._uses_cumulative_tool_stream_state(
        _build_responses_request(
            tool_choice={"type": "function", "name": "get_weather"}
        )
    )
    assert not parser._uses_cumulative_tool_stream_state(
        _build_responses_request(tool_choice="auto")
    )


def test_engine_stream_state_accumulates_only_when_requested() -> None:
    """Engine parsers retain normal delta behavior unless JSON needs history."""
    state = StreamState(engine_based=True)
    assert state.advance("first", [1]) == ("first", [1])
    state.commit("first", [1])
    assert state.previous_text == ""

    assert state.advance("second", [2], cumulative=True) == ("second", [2])
    state.commit("second", [2], cumulative=True)
    assert state.advance("third", [3], cumulative=True) == ("secondthird", [2, 3])


def test_gemma4_keeps_special_tokens_with_tools_thinking_disabled() -> None:
    """tools active + thinking disabled: ``skip_special_tokens`` must stay
    False so ``<|tool_call>`` delimiters reach the extractor. The merged
    enable_thinking early-return stripped them, breaking tool calling when
    thinking is off.
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(
        tool_choice="auto", chat_template_kwargs={"enable_thinking": False}
    )

    parser.adjust_request(request)

    assert request.skip_special_tokens is False


def test_gemma4_strips_special_tokens_when_nothing_to_preserve() -> None:
    """No active tools + thinking disabled: keep the default
    (``skip_special_tokens=True``) so stray delimiters do not leak into
    content.
    """
    parser = Gemma4ToolParser(_StubTokenizer())
    request = _build_chat_request(
        tool_choice="none", chat_template_kwargs={"enable_thinking": False}
    )

    parser.adjust_request(request)

    assert request.skip_special_tokens is True
