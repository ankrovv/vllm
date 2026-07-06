# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai.types.responses.tool import Tool
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.logger import init_logger

logger = init_logger(__name__)

CONTENT_NULL_STOP_REASON = "content_null"


class HarmonyTerminalState(Enum):
    """Outcome of classifying one completed Harmony choice."""

    TOOL_CALLS = "tool_calls"
    CONTENT = "content"
    RECOVERED_CONTENT = "recovered_content"
    CONTENT_NULL = "content_null"


@dataclass(frozen=True)
class HarmonyTerminalResult:
    state: HarmonyTerminalState
    content: str | None = None


def is_visible_content_empty(content: str | None) -> bool:
    """Return whether content is absent or only whitespace."""

    return content is None or not content.strip()


def is_function_recipient(
    recipient: str,
    allowed_function_tool_names: frozenset[str] | None = None,
) -> bool:
    """Check whether *recipient* refers to a function tool call.

    The optional *allowed_function_tool_names* parameter is used by the
    Responses API to distinguish bare function-call recipients (missing the
    ``functions.`` prefix) from MCP tool calls.  When provided, a bare
    recipient is only treated as a function call if it appears in the set.
    The Chat Completions path omits this parameter so that all bare
    recipients are accepted as function calls (the heuristic fallback).
    """
    if not recipient:
        return False
    if recipient.startswith("<|"):
        return False
    if recipient.startswith("functions."):
        return len(recipient) > len("functions.")
    if recipient == "assistant":
        return False
    if recipient in BUILTIN_TOOL_TO_MCP_SERVER_LABEL:
        return False
    first_segment = recipient.split(".", 1)[0]
    if first_segment in BUILTIN_TOOL_TO_MCP_SERVER_LABEL:
        return False
    if allowed_function_tool_names is not None:
        return recipient in allowed_function_tool_names
    return True


def extract_function_from_recipient(recipient: str) -> str:
    return recipient.removeprefix("functions.")


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_harmony_encoding = None

# Builtin tools that should be included in the system message when
# they are available and requested by the user.
# Tool args are provided by MCP tool descriptions. Output
# of the tools are stringified.
BUILTIN_TOOL_TO_MCP_SERVER_LABEL: dict[str, str] = {
    "python": "code_interpreter",
    "browser": "web_search_preview",
    "container": "container",
}

# Derive MCP_BUILTIN_TOOLS from the canonical mapping
MCP_BUILTIN_TOOLS: set[str] = set(BUILTIN_TOOL_TO_MCP_SERVER_LABEL.values())


def has_custom_tools(tool_types: set[str]) -> bool:
    """
    Checks if the given tool types are custom tools
    (i.e. any tool other than MCP builtin tools)
    """
    return not tool_types.issubset(MCP_BUILTIN_TOOLS)


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_system_message(
    model_identity: str | None = None,
    reasoning_effort: str | None = None,
    start_date: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    instructions: str | None = None,
    with_custom_tools: bool = False,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if instructions is not None and envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        current_identity = sys_msg_content.model_identity
        new_identity = (
            f"{current_identity}\n{instructions}" if current_identity else instructions
        )
        sys_msg_content = sys_msg_content.with_model_identity(new_identity)
    if reasoning_effort is not None:
        if reasoning_effort not in REASONING_EFFORT:
            supported_values = ", ".join(REASONING_EFFORT)
            raise ValueError(
                f"reasoning_effort={reasoning_effort!r} is not supported by "
                f"Harmony. Supported values are: {supported_values}."
            )
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort]
        )
    if start_date is None:
        # NOTE(woosuk): This brings non-determinism in vLLM.
        # Set VLLM_SYSTEM_START_DATE to pin it.
        start_date = envs.VLLM_SYSTEM_START_DATE or datetime.datetime.now().strftime(
            "%Y-%m-%d"
        )
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    if container_description is not None:
        sys_msg_content = sys_msg_content.with_tools(container_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def create_tool_definition(tool: ChatCompletionToolsParam | Tool):
    if isinstance(tool, ChatCompletionToolsParam):
        return ToolDescription.new(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=tool.function.parameters,
        )
    return ToolDescription.new(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.parameters,
    )


def get_developer_message(
    instructions: str | None = None,
    tools: list[Tool | ChatCompletionToolsParam] | None = None,
    *,
    force_instructions: bool = False,
) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None and (
        force_instructions or not envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS
    ):
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools: list[Tool | ChatCompletionToolsParam] = []
        for tool in tools:
            if tool.type in (
                "web_search_preview",
                "code_interpreter",
                "container",
            ):
                pass

            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.type} not supported")
        if function_tools:
            function_tool_descriptions = [
                create_tool_definition(tool) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions
            )
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def get_system_or_developer_message(role: str, instructions: str) -> Message:
    if role == "system" and envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        return get_system_message(instructions=instructions)
    return get_developer_message(instructions=instructions)


def parse_chat_inputs_to_harmony_messages(chat_msgs: list) -> list[Message]:
    """
    Parse a list of messages from request.messages in the Chat Completion API to
    Harmony messages.
    """
    msgs: list[Message] = []
    tool_id_names: dict[str, str] = {}

    # Collect tool id to name mappings for tool response recipient values
    for chat_msg in chat_msgs:
        for tool_call in chat_msg.get("tool_calls", []):
            tool_id_names[tool_call.get("id")] = tool_call.get("function", {}).get(
                "name"
            )

    for chat_msg in chat_msgs:
        msgs.extend(parse_chat_input_to_harmony_message(chat_msg, tool_id_names))

    msgs = auto_drop_analysis_messages(msgs)
    return msgs


def auto_drop_analysis_messages(msgs: list[Message]) -> list[Message]:
    """
    Harmony models expect the analysis messages (representing raw chain of thought) to
    be dropped after an assistant message to the final channel is produced from the
    reasoning of those messages.

    The openai-harmony library does this if the very last assistant message is to the
    final channel, but it does not handle the case where we're in longer multi-turn
    conversations and the client gave us reasoning content from previous turns of
    the conversation with multiple assistant messages to the final channel in the
    conversation.

    So, we find the index of the last assistant message to the final channel and drop
    all analysis messages that precede it, leaving only the analysis messages that
    are relevant to the current part of the conversation.
    """
    last_assistant_final_index = -1
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if msg.author.role == "assistant" and msg.channel == "final":
            last_assistant_final_index = i
            break

    cleaned_msgs: list[Message] = []
    for i, msg in enumerate(msgs):
        if i < last_assistant_final_index and msg.channel == "analysis":
            continue
        cleaned_msgs.append(msg)

    return cleaned_msgs


def flatten_input_text_content(content: Any) -> str | None:
    """
    Extract text parts from a Chat Completion or Responses API content field and
    flatten them into a single string. Returns None if no text content is found.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if text is not None:
                texts.append(text)
    return "".join(texts) if texts else None


def extract_instructions_from_messages(
    messages: Sequence[Any],
) -> tuple[str | None, list[Any]]:
    """
    Peel a leading system/developer Chat Completion or Responses message and
    flatten its instruction text.
    """
    remaining_messages = list(messages)
    if not remaining_messages:
        return None, remaining_messages

    first_message = remaining_messages[0]
    if not isinstance(first_message, dict):
        if hasattr(first_message, "to_dict"):
            # Handle OpenAI Harmony Message
            first_message = first_message.to_dict()
        elif hasattr(first_message, "model_dump"):
            first_message = first_message.model_dump(exclude_none=True)
        else:
            raise ValueError(f"Unknown message type: {type(first_message)}")

    if first_message.get("role") not in (
        "system",
        "developer",
    ):
        return None, remaining_messages

    instructions = flatten_input_text_content(first_message.get("content"))
    return instructions, remaining_messages[1:]


def build_harmony_preamble(
    *,
    instructions: str | None = None,
    tools: list[Tool | ChatCompletionToolsParam] | None = None,
    reasoning_effort: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    with_custom_tools: bool = False,
    force_developer_instructions: bool = False,
) -> list[Message]:
    """
    Build the standard Harmony system/developer prefix for a request.
    """
    developer_instructions = system_instructions = None
    if force_developer_instructions:
        developer_instructions = instructions
    elif envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        system_instructions = instructions
    else:
        developer_instructions = instructions

    messages = [
        get_system_message(
            reasoning_effort=reasoning_effort,
            browser_description=browser_description,
            python_description=python_description,
            container_description=container_description,
            instructions=system_instructions,
            with_custom_tools=with_custom_tools,
        )
    ]
    if developer_instructions or tools:
        messages.append(
            get_developer_message(
                instructions=developer_instructions,
                tools=tools,
                force_instructions=force_developer_instructions,
            )
        )
    return messages


def parse_chat_input_to_harmony_message(
    chat_msg, tool_id_names: dict[str, str] | None = None
) -> list[Message]:
    """
    Parse a message from request.messages in the Chat Completion API to
    Harmony messages.
    """
    tool_id_names = tool_id_names or {}

    if not isinstance(chat_msg, dict):
        # Handle Pydantic models
        chat_msg = chat_msg.model_dump(exclude_none=True)

    role = chat_msg.get("role")
    msgs: list[Message] = []

    # Assistant message with tool calls
    tool_calls = chat_msg.get("tool_calls", [])

    if role == "assistant" and tool_calls:
        content = flatten_input_text_content(chat_msg.get("content"))
        if content:
            commentary_msg = Message.from_role_and_content(Role.ASSISTANT, content)
            commentary_msg = commentary_msg.with_channel("commentary")
            msgs.append(commentary_msg)

        reasoning = chat_msg.get("reasoning")
        if reasoning:
            analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
            analysis_msg = analysis_msg.with_channel("analysis")
            msgs.append(analysis_msg)

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "") or ""
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            # Officially, this should be `<|constrain|>json` but there is not clear
            # evidence that improves accuracy over `json` and some anecdotes to the
            # contrary. Further testing of the different content_types is needed.
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    # Tool role message (tool output)
    if role == "tool":
        tool_call_id = chat_msg.get("tool_call_id", "")
        name = tool_id_names.get(tool_call_id, "")
        content = flatten_input_text_content(chat_msg.get("content")) or ""

        msg = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, f"functions.{name}"), content
            )
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        return [msg]

    # Non-tool reasoning content
    reasoning = chat_msg.get("reasoning")
    if role == "assistant" and reasoning:
        analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
        analysis_msg = analysis_msg.with_channel("analysis")
        msgs.append(analysis_msg)

    # Default: user/assistant/system messages with content
    content = chat_msg.get("content") or ""
    if content is None:
        content = ""
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.get("text", "")) for c in content]

    # Only add assistant messages if they have content, as reasoning or tool calling
    # assistant messages were already added above.
    if role == "assistant" and contents and contents[0].text:
        msg = Message.from_role_and_contents(role, contents)
        # Send non-tool assistant messages to the final channel
        msg = msg.with_channel("final")
        msgs.append(msg)
    elif role in ("system", "developer"):
        instructions = flatten_input_text_content(chat_msg.get("content"))
        if instructions is not None:
            msg = get_system_or_developer_message(role, instructions)
            msgs.append(msg)
    # For user messages, add them directly even if no content.
    elif role != "assistant":
        msg = Message.from_role_and_contents(role, contents)
        msgs.append(msg)

    return msgs


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    return token_ids


class _HarmonyControlTokens:
    """Cached Harmony control-token ids used to repair a missing delimiter.

    Resolved lazily from the encoding so there are no magic token numbers.
    """

    def __init__(self) -> None:
        enc = get_encoding()
        self._enc = enc

        def tid(text: str) -> int:
            return enc.encode(text, allowed_special="all")[0]

        self.channel = tid("<|channel|>")
        self.message = tid("<|message|>")
        self.constrain = tid("<|constrain|>")
        self.recipient = tid(" to")  # start of a ` to=<recipient>` tool target
        self.start = tid("<|start|>")
        self.end = tid("<|end|>")
        self.return_ = tid("<|return|>")
        self.call = tid("<|call|>")
        # Tokens that terminate or restart a header; content never starts with one.
        self.breakers = {
            tid("<|channel|>"),
            self.start,
            self.end,
            self.return_,
            self.call,
        }
        # Channel names whose content is user-visible (so a dropped body matters),
        # as their token sequences (e.g. ``commentary`` is two tokens).
        self.visible_channel_token_seqs = {
            tuple(enc.encode("final", allowed_special="all")),
            tuple(enc.encode("commentary", allowed_special="all")),
        }

    def is_whitespace(self, token_id: int) -> bool:
        try:
            return self._enc.decode([token_id]).strip() == ""
        except Exception:
            return False


_harmony_control: _HarmonyControlTokens | None = None


def _get_harmony_control() -> _HarmonyControlTokens:
    global _harmony_control
    if _harmony_control is None:
        _harmony_control = _HarmonyControlTokens()
    return _harmony_control


def _matches_at(
    token_ids: Sequence[int], start: int, expected: tuple[int, ...]
) -> bool:
    end = start + len(expected)
    return end <= len(token_ids) and tuple(token_ids[start:end]) == expected


def _header_has_recipient(
    token_ids: Sequence[int], channel_index: int, message_index: int
) -> bool:
    """Conservatively detect a tool recipient around a channel header."""

    ctrl = _get_harmony_control()
    header_start = 0
    for index in range(channel_index - 1, -1, -1):
        if token_ids[index] in ctrl.breakers:
            header_start = index + 1
            break
    return ctrl.recipient in token_ids[header_start:message_index]


def _extract_complete_json(text: str, *, allow_metadata_prefix: bool) -> str | None:
    """Return one complete JSON object/array surrounded only by allowed metadata."""

    decoder = json.JSONDecoder()
    harmless_wrapper_chars = frozenset(" \t\r\n()|`")
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        prefix = text[:index]
        suffix = text[index + end :]
        if any(char not in harmless_wrapper_chars for char in suffix):
            continue
        if not allow_metadata_prefix and any(
            char not in harmless_wrapper_chars for char in prefix
        ):
            continue
        return text[index : index + end]
    return None


def _extract_json_from_missing_body(text: str) -> str | None:
    """Recover JSON while rejecting identifier-like channel continuations."""

    recovered = _extract_complete_json(text, allow_metadata_prefix=True)
    if recovered is None:
        return None
    prefix = text[: text.find(recovered)]
    harmless_wrapper_chars = frozenset(' \t\r\n()|`".')
    if all(char in harmless_wrapper_chars for char in prefix):
        return recovered
    if prefix[:1].isspace() and not any(
        char.isascii() and (char.isalnum() or char == "_") for char in prefix
    ):
        return recovered
    return None


def _is_plausible_missing_message_body(text: str) -> bool:
    """Reject channel-name continuations such as ``final_output``."""

    return bool(text) and (text[0].isspace() or text[0] in '{[(.|`"')


def recover_harmony_visible_content(token_ids: Sequence[int]) -> str | None:
    """Recover only unambiguously user-visible Harmony message bodies.

    This is intentionally narrower than decoding the raw model output. It accepts
    exact ``final`` and recipient-less ``commentary`` channel headers, optionally
    with a ``<|constrain|>...`` content type. It decodes only the delimited body,
    or the body up to a terminal control token when ``<|message|>`` is missing.
    Analysis, tool recipients, unknown channels, ambiguous nested headers, and
    Harmony control tokens are never returned.
    """

    ctrl = _get_harmony_control()
    recovered: list[str] = []

    for channel_index, token_id in enumerate(token_ids):
        if token_id != ctrl.channel:
            continue

        for channel_tokens in ctrl.visible_channel_token_seqs:
            channel_start = channel_index + 1
            if not _matches_at(token_ids, channel_start, channel_tokens):
                continue

            cursor = channel_start + len(channel_tokens)
            while cursor < len(token_ids) and ctrl.is_whitespace(token_ids[cursor]):
                cursor += 1
            if cursor >= len(token_ids):
                continue

            constrained_without_delimiter = False
            if token_ids[cursor] == ctrl.constrain:
                cursor += 1
                constrained_start = cursor
                while cursor < len(token_ids) and token_ids[cursor] != ctrl.message:
                    if token_ids[cursor] in ctrl.breakers:
                        break
                    cursor += 1
                constrained_without_delimiter = (
                    cursor >= len(token_ids) or token_ids[cursor] != ctrl.message
                )

                if constrained_without_delimiter:
                    if _header_has_recipient(
                        token_ids, channel_index, constrained_start
                    ):
                        continue
                    raw_candidate = ctrl._enc.decode(
                        token_ids[constrained_start:cursor]
                    )
                    recovered_json = _extract_complete_json(
                        raw_candidate, allow_metadata_prefix=True
                    )
                    if recovered_json is not None:
                        recovered.append(recovered_json)
                    break

            if cursor >= len(token_ids):
                continue
            if token_ids[cursor] == ctrl.recipient:
                continue
            has_message_delimiter = token_ids[cursor] == ctrl.message
            if not has_message_delimiter and token_ids[cursor] in ctrl.breakers:
                continue
            if _header_has_recipient(token_ids, channel_index, cursor):
                continue

            body_start = cursor + 1 if has_message_delimiter else cursor
            body_end = body_start
            while (
                body_end < len(token_ids) and token_ids[body_end] not in ctrl.breakers
            ):
                body_end += 1

            body_tokens = token_ids[body_start:body_end]
            # Nested header controls make the candidate ambiguous. Fail closed.
            if ctrl.message in body_tokens or ctrl.constrain in body_tokens:
                continue

            body = ctrl._enc.decode(body_tokens)
            if not has_message_delimiter:
                if not _is_plausible_missing_message_body(body):
                    continue
                recovered_json = _extract_json_from_missing_body(body)
                if recovered_json is not None:
                    body = recovered_json
                elif (
                    _extract_complete_json(body, allow_metadata_prefix=True) is not None
                ):
                    continue
            if not is_visible_content_empty(body):
                recovered.append(body)
            break

    return "\n".join(recovered) or None


def classify_harmony_terminal(
    *,
    content: str | None,
    has_tool_calls: bool,
    token_ids: Sequence[int],
    allow_recovery: bool = True,
) -> HarmonyTerminalResult:
    """Classify a terminal Harmony choice without exposing hidden channels."""

    if has_tool_calls:
        return HarmonyTerminalResult(HarmonyTerminalState.TOOL_CALLS, content)
    if not is_visible_content_empty(content):
        return HarmonyTerminalResult(HarmonyTerminalState.CONTENT, content)
    if allow_recovery:
        recovered = recover_harmony_visible_content(token_ids)
        if not is_visible_content_empty(recovered):
            return HarmonyTerminalResult(
                HarmonyTerminalState.RECOVERED_CONTENT, recovered
            )
    return HarmonyTerminalResult(HarmonyTerminalState.CONTENT_NULL, "")


class _RepairingStreamableParser(StreamableParser):
    """``StreamableParser`` that repairs a visible channel header emitted without
    its ``<|message|>`` delimiter.

    GPT-OSS occasionally emits a user-visible channel header (``<|channel|>final``
    or ``<|channel|>commentary``) directly followed by the message body, omitting
    the required ``<|message|>`` delimiter. The base parser then never leaves the
    header state and the generated answer is silently dropped (the response
    returns ``content=None`` even though the tokens were generated and billed).

    This subclass watches the token stream and defers only a malformed visible
    header until it sees either a real delimiter or the message terminator. In the
    latter case it inserts the missing ``<|message|>`` and replays the buffered body
    through the normal content path. It is a no-op on well-formed output. Because
    every consumer -- streaming chat, non-streaming chat, and the Responses API --
    builds its parser through ``get_streamable_parser_for_assistant``, repairing here
    covers them all.
    """

    def __init__(self, encoding: Any, role: Any, *, strict: bool = True) -> None:
        super().__init__(encoding, role, strict=strict)
        self._ctrl = _get_harmony_control()
        # State of the small header-tracking machine.
        self._reading_channel_name = False  # just saw ``<|channel|>``
        self._channel_name: tuple[int, ...] = ()
        self._awaiting_delimiter = False  # in a visible header, no delimiter yet
        self._pending_visible_tokens: list[int] = []
        self._awaiting_constrained_delimiter = False
        self._pending_constraint_tokens: list[int] = []

    def process(self, token: int) -> "StreamableParser":
        ctrl = self._ctrl
        if self._awaiting_constrained_delimiter:
            if token == ctrl.message:
                # The body delimiter is unambiguous. Discard malformed content-
                # type metadata and let the base parser consume the body normally.
                self._pending_constraint_tokens.clear()
                self._awaiting_constrained_delimiter = False
                return super().process(token)
            if token in ctrl.breakers:
                # No body delimiter appeared. Leave the parser at the exact visible
                # channel and let the terminal classifier inspect the raw tokens.
                self._pending_constraint_tokens.clear()
                self._awaiting_constrained_delimiter = False
                return self
            self._pending_constraint_tokens.append(token)
            return self
        if self._awaiting_delimiter:
            if token == ctrl.message:
                # A real delimiter wins. Anything buffered between the exact
                # channel name and this delimiter was malformed header metadata,
                # not message content.
                self._pending_visible_tokens.clear()
                self._awaiting_delimiter = False
                return super().process(token)
            if token == ctrl.constrain:
                self._pending_visible_tokens.clear()
                self._awaiting_delimiter = False
                self._awaiting_constrained_delimiter = True
                self._pending_constraint_tokens = []
                return self
            if token == ctrl.recipient:
                self._pending_visible_tokens.clear()
                self._awaiting_delimiter = False
                return super().process(token)
            if token in ctrl.breakers:
                pending = self._pending_visible_tokens
                self._pending_visible_tokens = []
                self._awaiting_delimiter = False
                if any(not ctrl.is_whitespace(item) for item in pending):
                    pending_text = ctrl._enc.decode(pending)
                    if not _is_plausible_missing_message_body(pending_text):
                        return super().process(token)
                    recovered_json = _extract_json_from_missing_body(pending_text)
                    if recovered_json is not None:
                        try:
                            pending = ctrl._enc.encode(recovered_json)
                        except ValueError:
                            # A literal Harmony control marker inside the JSON is
                            # ambiguous. Leave no content for the terminal guard.
                            pending = []
                    elif (
                        _extract_complete_json(pending_text, allow_metadata_prefix=True)
                        is not None
                    ):
                        return super().process(token)
                    # No real delimiter appeared before the message ended. The
                    # buffered tokens are the body, so replay them through the
                    # normal content state after injecting ``<|message|>``.
                    if pending:
                        super().process(ctrl.message)
                        for item in pending:
                            super().process(item)
                return super().process(token)
            # Delay only a malformed visible header. This lookahead prevents a
            # header such as ``final JSON<|message|>...`` from being mistaken for
            # content while leaving well-formed output fully streaming.
            self._pending_visible_tokens.append(token)
            return self
        if self._reading_channel_name:
            self._channel_name += (token,)
            name = self._channel_name
            seqs = ctrl.visible_channel_token_seqs
            if name in seqs:
                self._reading_channel_name = False
                self._awaiting_delimiter = True
                self._pending_visible_tokens = []
            elif not any(seq[: len(name)] == name for seq in seqs):
                self._reading_channel_name = False  # analysis / unrecognized
            return super().process(token)
        if token == ctrl.channel:
            self._reading_channel_name = True
            self._channel_name = ()
        return super().process(token)


def get_streamable_parser_for_assistant() -> StreamableParser:
    return _RepairingStreamableParser(get_encoding(), role=Role.ASSISTANT)


def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        try:
            parser.process(token_id)
        except HarmonyError:
            logger.warning(
                "Harmony parsing failed; deferring to filtered terminal recovery."
            )
            return get_streamable_parser_for_assistant()
    return parser


def parse_chat_output(
    token_ids: Sequence[int],
) -> tuple[str | None, str | None, bool]:
    """
    Parse the output of a Harmony chat completion into reasoning and final content.
    Note that when the `openai` tool parser is used, serving_chat only uses this
    for the reasoning content and gets the final content from the tool call parser.

    When the `openai` tool parser is not enabled, or when `GptOssReasoningParser` is
    in use,this needs to return the final content without any tool calls parsed.

    Empty reasoning or final content is returned as None instead of an empty string.
    """
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    is_tool_call = False  # TODO: update this when tool call is supported

    # Get completed messages from the parser
    # - analysis channel: hidden reasoning
    # - commentary channel without recipient (preambles): visible to user
    # - final channel: visible to user
    # - commentary with recipient (tool calls): handled separately by tool parser
    reasoning_texts = [
        msg.content[0].text for msg in output_msgs if msg.channel == "analysis"
    ]
    final_texts = [
        msg.content[0].text
        for msg in output_msgs
        if msg.channel == "final" or (msg.channel == "commentary" and not msg.recipient)
    ]

    # Extract partial messages from the parser
    if parser.current_channel == "analysis" and parser.current_content:
        reasoning_texts.append(parser.current_content)
    elif parser.current_channel == "final" and parser.current_content:
        final_texts.append(parser.current_content)
    elif (
        parser.current_channel == "commentary"
        and not parser.current_recipient
        and parser.current_content
    ):
        # Preambles (commentary without recipient) are visible to user
        final_texts.append(parser.current_content)

    # Flatten multiple messages into a single string
    reasoning: str | None = "\n".join(reasoning_texts)
    final_content: str | None = "\n".join(final_texts)

    # Return None instead of empty string since existing callers check for None
    reasoning = reasoning or None
    final_content = final_content or None

    return reasoning, final_content, is_tool_call
