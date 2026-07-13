# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import Gemma4ParserToolAdapter


class Gemma4EngineToolParser(Gemma4ParserToolAdapter):  # type: ignore[valid-type, misc]
    """Gemma 4 parser with enforced required and named tool choice.

    Automatic tool choice keeps Gemma's native tool-call format. Required and
    named tool choice use the standard JSON constraint and serving-layer
    extraction.
    """

    supports_required_and_named = True
    # Standard required/named streaming parses progressively accumulated JSON.
    # This applies only to forced choices; auto keeps the native delta parser.
    cumulative_tool_streaming_for_required_and_named = True
