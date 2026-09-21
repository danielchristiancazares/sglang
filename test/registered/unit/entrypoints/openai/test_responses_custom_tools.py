import asyncio
import unittest
from unittest.mock import Mock

from utils import (
    StreamFixture,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.responses_adapters import (
    decode_custom_tool_input,
    decode_custom_tool_input_prefix,
    decode_reasoning_state,
    encode_custom_tool_input,
    encode_reasoning_state,
    label_developer_content,
    response_tool_declarations,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CUSTOM_TOOL = {
    "type": "custom",
    "name": "emit_command",
    "description": "Emit a shell command.",
    "format": {"type": "text"},
}


def _custom_request(**kwargs) -> ResponsesRequest:
    payload = {
        "model": "x",
        "input": "run pwd",
        "tools": [CUSTOM_TOOL],
        "tool_choice": "required",
        "store": False,
    }
    payload.update(kwargs)
    return ResponsesRequest(**payload)


class CustomToolAdapterTestCase(CustomTestCase):
    def test_payload_survives_encode_decode(self):
        for payload in ("pwd", 'echo "hi"', "a\nb\tc", "naïve 😀", "{not json}"):
            self.assertEqual(
                decode_custom_tool_input(encode_custom_tool_input(payload)), payload
            )
        self.assertEqual(decode_custom_tool_input("pwd"), "pwd")

    def test_prefix_decode_tracks_a_growing_buffer(self):
        payload = 'echo "a\nb" 😀'
        arguments = encode_custom_tool_input(payload)
        seen = ""
        for end in range(len(arguments) + 1):
            prefix = decode_custom_tool_input_prefix(arguments[:end])
            # Monotonic, and never runs ahead of the finished value.
            self.assertTrue(prefix.startswith(seen), (seen, prefix))
            self.assertTrue(payload.startswith(prefix), (payload, prefix))
            seen = prefix
        self.assertEqual(seen, payload)
        self.assertEqual(decode_custom_tool_input_prefix('{"city": "Beijing"}'), "")


class CustomToolShimTestCase(CustomTestCase):
    def test_custom_tool_becomes_a_single_string_function_tool(self):
        request = _custom_request()
        (tool,) = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertEqual(tool.function.name, "emit_command")
        self.assertEqual(list(tool.function.parameters["properties"]), ["input"])
        self.assertEqual(tool.function.parameters["required"], ["input"])

    def test_grammar_format_is_described_to_the_model(self):
        request = _custom_request(
            tools=[
                {
                    **CUSTOM_TOOL,
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": 'start: "pwd"',
                    },
                }
            ]
        )
        (tool,) = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertIn("lark", tool.function.description)
        self.assertIn('start: "pwd"', tool.function.description)

    def test_required_tool_choice_accepts_a_custom_tool(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = _custom_request()
        output_items = serving._make_response_output_items(
            request,
            '[{"name": "emit_command", "parameters": {"input": "pwd"}}]',
            tokenizer=Mock(),
            require_reasoning=False,
        )
        (item,) = output_items
        self.assertEqual(item.type, "custom_tool_call")
        self.assertEqual(item.name, "emit_command")
        self.assertEqual(item.input, "pwd")
        self.assertTrue(item.call_id)

    def test_named_choice_parses_json_in_full_and_stream_responses(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        for tool_type in ("function", "custom"):
            for nested in (False, True):
                with self.subTest(tool_type=tool_type, nested=nested):
                    name = "emit_command"
                    choice = {"type": tool_type}
                    choice.update(
                        {"function": {"name": name}} if nested else {"name": name}
                    )
                    request = _custom_request(
                        tools=[{"type": tool_type, "name": name}],
                        tool_choice=choice,
                        stream=True,
                    )
                    raw = '[{"name":"emit_command","parameters":{"input":"pwd"}}]'
                    (item,) = serving._make_response_output_items(
                        request, raw, tokenizer=Mock(), require_reasoning=False
                    )
                    self.assertEqual(
                        item.type,
                        f"{tool_type}_tool_call"
                        if tool_type == "custom"
                        else "function_call",
                    )
                    events = StreamFixture(serving, request).run(
                        [engine_chunk(raw[:30]), engine_chunk(raw, 2, finish=True)]
                    )
                    (stream_item,) = find_completed_event(events)["response"]["output"]
                    self.assertEqual(stream_item["type"], item.type)
                    self.assertEqual(stream_item["name"], name)
                    field = "input" if tool_type == "custom" else "arguments"
                    self.assertEqual(stream_item[field], getattr(item, field))
                    self.assertNotIn("response.output_text.delta", event_types(events))

    def test_named_choice_rejects_an_undeclared_tool_before_generation(self):
        serving = make_serving()
        for stream in (False, True):
            request = _custom_request(
                tool_choice={"type": "custom", "name": "missing"}, stream=stream
            )
            result = asyncio.run(serving.create_responses(request))
            self.assertEqual(result.status_code, 400)
            self.assertIn(b"tool_choice", result.body)
        serving.tokenizer_manager.generate_request.assert_not_called()


class NamespacedToolAdapterTestCase(CustomTestCase):
    def request(self) -> ResponsesRequest:
        return ResponsesRequest(
            model="x",
            input="Create ready.txt and inspect it.",
            tools=[
                {"type": "custom", "name": "exec", "format": {"type": "text"}},
                {
                    "type": "namespace",
                    "name": "workspace",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "description": "Apply the supplied patch.",
                            "format": {"type": "text"},
                        },
                        {
                            "type": "function",
                            "name": "inspect",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    ],
                },
            ],
        )

    def test_all_declared_tools_reach_the_chat_template(self):
        tools = OpenAIServingResponses._response_tools_to_chat_tools(self.request())
        self.assertEqual(
            [tool.function.name for tool in tools],
            ["exec", "workspace.apply_patch", "workspace.inspect"],
        )
        self.assertEqual(tools[1].function.parameters["required"], ["input"])
        self.assertEqual(tools[2].function.parameters["required"], ["path"])

    def test_custom_call_preserves_namespace_through_streaming_and_replay(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        request = self.request()
        payload = "*** Begin Patch\n*** Add File: ready.txt\n+READY 😀\n*** End Patch"
        raw = (
            "<tool_call>\n<function=workspace.apply_patch>\n<parameter=input>\n"
            + payload + "\n</parameter>\n</function>\n</tool_call>"
        )
        events = StreamFixture(serving, request).run([
            engine_chunk(raw[:11]),
            engine_chunk(raw[:56], 2),
            engine_chunk(raw[:-9], 3),
            engine_chunk(raw, 4, finish=True),
        ])
        final = find_completed_event(events)["response"]
        (call,) = final["output"]
        self.assertEqual(call["type"], "custom_tool_call")
        self.assertEqual(call["name"], "apply_patch")
        self.assertEqual(call["namespace"], "workspace")
        self.assertEqual(call["input"], payload)
        deltas = "".join(
            item["delta"] for item in event_payloads(events)
            if item["type"] == "response.custom_tool_call_input.delta"
        )
        self.assertEqual(deltas, payload)
        replay = serving._normalize_response_message_for_chat(call)
        self.assertEqual(replay["tool_calls"][0]["function"]["name"], "workspace.apply_patch")
        self.assertEqual(
            decode_custom_tool_input(replay["tool_calls"][0]["function"]["arguments"]),
            payload,
        )

    def test_function_call_preserves_namespace_in_full_response(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        raw = (
            "<tool_call>\n<function=workspace.inspect>\n"
            "<parameter=path>\nready.txt\n</parameter>\n</function>\n</tool_call>"
        )
        (call,) = serving._make_response_output_items(
            self.request(), raw, tokenizer=Mock(), require_reasoning=False,
        )
        wire = call.model_dump()
        self.assertEqual(wire["name"], "inspect")
        self.assertEqual(wire["namespace"], "workspace")
        replay = serving._normalize_response_message_for_chat(wire)
        self.assertEqual(replay["tool_calls"][0]["function"]["name"], "workspace.inspect")

    def test_named_choice_qualifies_the_namespace(self):
        request = self.request()
        request.tool_choice = {"type": "custom", "name": "apply_patch", "namespace": "workspace"}
        self.assertEqual(
            request.effective_tool_choice(),
            {"type": "function", "name": "workspace.apply_patch"},
        )

    def test_undeclared_calls_raise_a_protocol_failure(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        raw = "<tool_call><function=workspace.delete_all></function></tool_call>"
        with self.assertRaisesRegex(ValueError, "undeclared tool"):
            serving._make_response_output_items(
                self.request(), raw, tokenizer=Mock(), require_reasoning=False,
            )
        events = StreamFixture(serving, self.request()).run([
            engine_chunk(raw, 1, finish=True),
        ])
        self.assertIn("response.failed", event_types(events))
        self.assertNotIn("response.completed", event_types(events))

    def test_nested_namespaces_retain_the_complete_address(self):
        request = ResponsesRequest(
            model="x", input="Use the nested tool.",
            tools=[{
                "type": "namespace", "name": "outer",
                "tools":[{
                    "type": "namespace", "name": "inner",
                    "tools":[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
                }],
            }],
        )
        (tool,) = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertEqual(tool.function.name, "outer.inner.lookup")

    def test_empty_namespace_keeps_an_empty_callable_collection(self):
        request = ResponsesRequest(
            model="x", input="hi",
            tools=[{"type": "namespace", "name": "workspace", "tools": []}],
        )
        self.assertEqual(OpenAIServingResponses._response_tools_to_chat_tools(request), [])

    def test_malformed_and_ambiguous_namespaces_are_rejected_before_generation(self):
        declarations = (
            {"type": "namespace", "name": "workspace"},
            {"type": "namespace", "name": "workspace", "tools": [{"type": "function"}]},
            {"type": "namespace", "name": "workspace", "tools": [{"type": "web_search"}]},
        )
        for declaration in declarations:
            with self.subTest(declaration=declaration):
                serving = make_serving()
                request = ResponsesRequest(model="x", input="hi", tools=[declaration])
                response = asyncio.run(serving.create_responses(request))
                self.assertEqual(response.status_code, 400)
                serving.tokenizer_manager.generate_request.assert_not_called()
        request = self.request()
        request.tools.append(request.tools[0].model_copy())
        with self.assertRaisesRegex(ValueError, "ambiguous tool address"):
            response_tool_declarations(request.tools)


class CustomToolReplayTestCase(CustomTestCase):
    def test_custom_tool_call_replays_through_the_shim(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "emit_command",
                "input": "pwd",
            }
        )
        self.assertEqual(message["role"], "assistant")
        (call,) = message["tool_calls"]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"]["name"], "emit_command")
        self.assertEqual(decode_custom_tool_input(call["function"]["arguments"]), "pwd")

    def test_custom_tool_call_output_becomes_a_tool_message(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "/workspace",
            }
        )
        self.assertEqual(
            message,
            {"role": "tool", "tool_call_id": "call_1", "content": "/workspace"},
        )

        parts = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": [{"type": "output_text", "text": "/work"}, {"text": "space"}],
            }
        )
        self.assertEqual(parts["content"], "/workspace")


class CustomToolStreamTestCase(CustomTestCase):
    def _stream(self, chunks):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = _custom_request(stream=True)
        return StreamFixture(serving, request).run(chunks)

    def test_input_deltas_reconstruct_the_final_payload(self):
        emitted = '[{"name": "emit_command", "parameters": {"input": "pwd"}}]'
        chunks = [engine_chunk(emitted[:i], i) for i in range(1, len(emitted))] + [
            engine_chunk(emitted, len(emitted), finish=True)
        ]

        events = self._stream(chunks)
        pairs = list(zip(event_types(events), event_payloads(events)))
        deltas = "".join(
            p["delta"] for t, p in pairs if t == "response.custom_tool_call_input.delta"
        )
        done = [p for t, p in pairs if t == "response.custom_tool_call_input.done"]

        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["input"], "pwd")
        self.assertEqual(deltas, "pwd")

        added = [p for t, p in pairs if t == "response.output_item.added"]
        item_done = [p for t, p in pairs if t == "response.output_item.done"]
        self.assertEqual(len(added), 1)
        self.assertEqual(len(item_done), 1)
        self.assertEqual(added[0]["item"]["type"], "custom_tool_call")
        self.assertEqual(added[0]["item"]["id"], item_done[0]["item"]["id"])

        final = find_completed_event(events)["response"]
        (item,) = [i for i in final["output"] if i["type"] == "custom_tool_call"]
        self.assertEqual(item["name"], "emit_command")
        self.assertEqual(item["input"], "pwd")
        self.assertTrue(item["call_id"])
        self.assertNotIn(
            "response.function_call_arguments.delta", [t for t, _ in pairs]
        )


GLM47_CALL = (
    "<tool_call>emit_command"
    "<arg_key>input</arg_key><arg_value>pwd</arg_value>"
    "</tool_call>"
)


class CustomToolGlm47FormatTestCase(CustomTestCase):
    """The shim has to survive a real model-native tool-call format, not just the
    JSON array the ``required`` constraint produces."""

    def _serving(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "glm47"
        return serving

    def test_non_streaming_glm47_call_becomes_a_custom_tool_call(self):
        serving = self._serving()
        request = _custom_request(tool_choice="auto")
        (item,) = serving._make_response_output_items(
            request, GLM47_CALL, tokenizer=Mock(), require_reasoning=False
        )
        self.assertEqual(item.type, "custom_tool_call")
        self.assertEqual(item.name, "emit_command")
        self.assertEqual(item.input, "pwd")

    def test_streaming_glm47_call_reconstructs_the_payload(self):
        serving = self._serving()
        request = _custom_request(tool_choice="auto", stream=True)
        chunks = [engine_chunk(GLM47_CALL[:i], i) for i in range(1, len(GLM47_CALL))]
        chunks.append(engine_chunk(GLM47_CALL, len(GLM47_CALL), finish=True))

        events = StreamFixture(serving, request).run(chunks)
        pairs = list(zip(event_types(events), event_payloads(events)))
        deltas = "".join(
            p["delta"] for t, p in pairs if t == "response.custom_tool_call_input.delta"
        )
        done = [p for t, p in pairs if t == "response.custom_tool_call_input.done"]

        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["input"], "pwd")
        self.assertEqual(deltas, done[0]["input"])

        final = find_completed_event(events)["response"]
        (item,) = [i for i in final["output"] if i["type"] == "custom_tool_call"]
        self.assertEqual(item["input"], "pwd")


class ReasoningEncryptedContentTestCase(CustomTestCase):
    def test_state_survives_encode_decode(self):
        for text in ("", "step one\nstep two", "naïve 😀"):
            self.assertEqual(decode_reasoning_state(encode_reasoning_state(text)), text)
        self.assertIsNone(decode_reasoning_state("not-ours"))
        self.assertIsNone(decode_reasoning_state(None))

    def test_reasoning_item_carries_the_blob_only_when_included(self):
        without = OpenAIServingResponses._make_reasoning_item(
            ResponsesRequest(model="x", input="hi", store=False),
            "because",
            item_id="rs_1",
            status=None,
        )
        self.assertIsNone(without.encrypted_content)

        with_blob = OpenAIServingResponses._make_reasoning_item(
            ResponsesRequest(
                model="x",
                input="hi",
                store=False,
                include=["reasoning.encrypted_content"],
            ),
            "because",
            item_id="rs_1",
            status=None,
        )
        self.assertEqual(decode_reasoning_state(with_blob.encrypted_content), "because")

    def test_blob_only_reasoning_item_replays(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "reasoning",
                "summary": [],
                "content": [],
                "encrypted_content": encode_reasoning_state("because the sky"),
            }
        )
        self.assertEqual(
            message, {"role": "assistant", "reasoning_content": "because the sky"}
        )

    def test_streamed_reasoning_item_carries_the_blob(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            include=["reasoning.encrypted_content"],
        )
        events = StreamFixture(serving, request, require_reasoning=True).run(
            [
                engine_chunk("because", 1),
                engine_chunk("because</think>answer", 2, finish=True),
            ]
        )
        final = find_completed_event(events)["response"]
        (item,) = [i for i in final["output"] if i["type"] == "reasoning"]
        self.assertEqual(decode_reasoning_state(item["encrypted_content"]), "because")

    def test_non_streaming_reasoning_item_carries_the_blob(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            include=["reasoning.encrypted_content"],
        )
        output_items = serving._make_response_output_items(
            request,
            "because</think>answer",
            tokenizer=Mock(),
            require_reasoning=True,
        )
        self.assertEqual(
            decode_reasoning_state(output_items[0].encrypted_content), "because"
        )


class DeveloperMessageTestCase(CustomTestCase):
    def test_content_is_labelled(self):
        self.assertEqual(
            label_developer_content("Be terse."),
            "Developer instructions:\nBe terse.",
        )
        self.assertEqual(
            label_developer_content(
                [{"type": "input_text", "text": "Be terse."}, {"type": "input_image"}]
            ),
            [
                {"type": "input_text", "text": "Developer instructions:\nBe terse."},
                {"type": "input_image"},
            ],
        )

    def test_developer_block_follows_instructions_in_the_system_message(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            store=False,
            instructions="Respond in English.",
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "Reply with exactly OK."}
                    ],
                },
                {"role": "user", "content": "Reply with exactly NO."},
            ],
        )
        messages = serving._construct_input_messages(request, None)
        self.assertEqual(
            messages[0],
            {
                "role": "system",
                "content": (
                    "Respond in English.\n\n"
                    "Developer instructions:\nReply with exactly OK."
                ),
            },
        )
        self.assertEqual(messages[1]["role"], "user")


class ModelValidationTestCase(CustomTestCase):
    def test_model_validation(self):
        serving = make_serving()
        error = serving._validate_model("__no_such_model__")
        self.assertIsNotNone(error)
        self.assertEqual(error.status_code, 404)
        self.assertIsNone(serving._validate_model(None))
        self.assertIsNone(serving._validate_model("x"))
        self.assertIsNone(serving._validate_model("x:my-adapter"))


if __name__ == "__main__":
    unittest.main()
