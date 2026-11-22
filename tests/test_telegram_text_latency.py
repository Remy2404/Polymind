import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from google.genai import errors, types

from src.ai.models.model_configs import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_CHAT_ALIAS,
    GEMINI_CHAT_FALLBACK_MODEL,
)
from src.ai.providers import gemini_api
from src.handlers.text import conversation
from src.services.rate_limiter import RateLimiter


@pytest.fixture
def chat(monkeypatch):
    sdk_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock()))
    )
    monkeypatch.setattr(gemini_api.genai, "Client", lambda **kwargs: sdk_client)
    monkeypatch.setattr(
        conversation, "get_mcp_tools_for_user", AsyncMock(return_value=[])
    )
    handler = conversation.TextConversationMixin()
    handler.gemini_api = gemini_api.GeminiAPI(RateLimiter(30), api_key="test-key")
    handler.openrouter_api = None
    handler.deepseek_api = None
    return handler, sdk_client.aio.models.generate_content


@pytest.fixture
def sdk_transport(monkeypatch):
    original_client = gemini_api.genai.Client

    def install(send):
        def create_client(**kwargs):
            return original_client(
                **kwargs,
                http_options=types.HttpOptions(
                    async_client_args={"transport": httpx.MockTransport(send)}
                ),
            )

        monkeypatch.setattr(gemini_api.genai, "Client", create_client)

    return install


def gemini_response(part):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
    )


def unavailable():
    return errors.ServerError(503, {"error": {"message": "overloaded"}})


def generate(handler, timeout=1):
    return asyncio.run(
        handler._generate_response(
            "Hello",
            [{"role": "user", "content": "Earlier"}],
            "Quote",
            GEMINI_CHAT_ALIAS,
            42,
            timeout,
        )
    )


def test_chat_success_uses_primary_once_and_preserves_context(chat):
    handler, sdk = chat
    sdk.return_value = gemini_response(types.Part.from_text(text="Hi"))

    assert generate(handler) == ("Hi", DEFAULT_GEMINI_MODEL)
    sdk.assert_awaited_once()
    request = sdk.call_args.kwargs
    assert request["model"] == DEFAULT_GEMINI_MODEL
    assert [part.text for part in request["contents"][0].parts] == [
        "Context:\nUser: Earlier",
        "Replying to: Quote",
        "Hello",
    ]
    assert request["config"].http_options.timeout == int(
        gemini_api.CHAT_PRIMARY_TIMEOUT_SECONDS * 1000
    )


def test_sdk_sends_timeout_and_fails_over_after_http_503(sdk_transport):
    requests = []

    async def send(request):
        requests.append(request)
        if request.url.path.endswith(f"/{DEFAULT_GEMINI_MODEL}:generateContent"):
            return httpx.Response(
                503,
                request=request,
                json={
                    "error": {
                        "code": 503,
                        "status": "UNAVAILABLE",
                        "message": "overload",
                    }
                },
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Hi"}]}}
                ]
            },
        )

    sdk_transport(send)
    api = gemini_api.GeminiAPI(RateLimiter(30), api_key="test-key")

    assert asyncio.run(api.generate_chat_response("Hello")) == (
        "Hi",
        GEMINI_CHAT_FALLBACK_MODEL,
    )
    assert [request.url.path.rsplit("/", 1)[1] for request in requests] == [
        "gemini-4-hydrogen:generateContent",
        f"{GEMINI_CHAT_FALLBACK_MODEL}:generateContent",
    ]
    assert [request.extensions["timeout"]["read"] for request in requests] == [
        gemini_api.CHAT_PRIMARY_TIMEOUT_SECONDS,
        gemini_api.CHAT_FALLBACK_TIMEOUT_SECONDS,
    ]
    primary, fallback = (json.loads(request.content) for request in requests)
    assert primary["contents"] == fallback["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]}
    ]
    sampling_fields = {"temperature", "topP", "topK"}
    assert sampling_fields.isdisjoint(primary["generationConfig"])
    assert sampling_fields <= fallback["generationConfig"].keys()


def test_sdk_preserves_thought_signature_in_tool_followup(sdk_transport, monkeypatch):
    signature = "cG9seW1pbmQ="
    requests = []

    async def send(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            part = {
                "functionCall": {"name": "lookup", "args": {}},
                "thoughtSignature": signature,
            }
        else:
            part = {"text": "42"}
        return httpx.Response(
            200, json={"candidates": [{"content": {"role": "model", "parts": [part]}}]}
        )

    sdk_transport(send)
    api = gemini_api.GeminiAPI(RateLimiter(30), api_key="test-key")
    execute = AsyncMock(return_value="42")
    monkeypatch.setattr(api, "_execute_mcp_tool", execute)
    tools = [{"function": {"name": "lookup", "description": "Look up a value"}}]

    assert asyncio.run(api.generate_chat_response("Hello", mcp_tools=tools)) == (
        "42",
        DEFAULT_GEMINI_MODEL,
    )

    execute.assert_awaited_once()
    model_content, tool_result = requests[1]["contents"][-2:]
    assert model_content["parts"][0]["thoughtSignature"] == signature
    assert tool_result["parts"][0]["functionResponse"] == {
        "name": "lookup",
        "response": {"result": "42"},
    }


def test_slow_primary_is_cancelled_before_fallback(chat, monkeypatch):
    handler, sdk = chat
    cancelled_models = []

    async def respond(*, model, **kwargs):
        if model == DEFAULT_GEMINI_MODEL:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled_models.append(model)
        return gemini_response(types.Part.from_text(text="Hi"))

    sdk.side_effect = respond
    monkeypatch.setattr(gemini_api, "CHAT_PRIMARY_TIMEOUT_SECONDS", 0.01)

    assert generate(handler) == ("Hi", GEMINI_CHAT_FALLBACK_MODEL)
    assert cancelled_models == [DEFAULT_GEMINI_MODEL]
    assert sdk.await_count == 2


@pytest.mark.parametrize("status, expected_attempts", [(400, 1), (503, 2)])
def test_provider_errors_do_not_trigger_extra_attempts(chat, status, expected_attempts):
    handler, sdk = chat
    error_type = errors.ClientError if status == 400 else errors.ServerError
    sdk.side_effect = error_type(status, {"error": {"message": "test failure"}})

    with pytest.raises(errors.APIError) as failure:
        generate(handler)

    assert failure.value.code == status
    assert sdk.await_count == expected_attempts


@pytest.mark.parametrize("followup_fails", [False, True])
def test_fallback_can_use_mcp_and_executes_tool_once(chat, monkeypatch, followup_fails):
    handler, sdk = chat
    tools = [{"function": {"name": "lookup", "description": "Look up a value"}}]
    monkeypatch.setattr(
        conversation, "get_mcp_tools_for_user", AsyncMock(return_value=tools)
    )
    execute = AsyncMock(return_value="42")
    monkeypatch.setattr(handler.gemini_api, "_execute_mcp_tool", execute)
    sdk.side_effect = [
        unavailable(),
        gemini_response(
            types.Part(function_call=types.FunctionCall(name="lookup", args={}))
        ),
        unavailable()
        if followup_fails
        else gemini_response(types.Part.from_text(text="42")),
    ]

    if followup_fails:
        with pytest.raises(errors.ServerError):
            generate(handler)
    else:
        assert generate(handler) == ("42", GEMINI_CHAT_FALLBACK_MODEL)

    execute.assert_awaited_once()
    assert [call.kwargs["model"] for call in sdk.await_args_list] == [
        DEFAULT_GEMINI_MODEL,
        GEMINI_CHAT_FALLBACK_MODEL,
        GEMINI_CHAT_FALLBACK_MODEL,
    ]
    initial, fallback, followup = [call.kwargs for call in sdk.await_args_list]
    assert initial["config"].tools == fallback["config"].tools
    assert followup["config"].tools is None
    assert followup["contents"][-1].parts[0].function_response.response == {
        "result": "42"
    }


def test_telegram_deadline_cancels_mcp_without_starting_plain_generation(
    chat, monkeypatch
):
    handler, _ = chat

    async def stalled_mcp(**kwargs):
        await asyncio.Event().wait()

    mcp = AsyncMock(side_effect=stalled_mcp)
    standard = Mock()
    monkeypatch.setattr(conversation, "is_model_mcp_compatible", lambda model: True)
    monkeypatch.setattr(conversation, "generate_mcp_response", mcp)
    monkeypatch.setattr(conversation.ModelHandlerFactory, "get_model_handler", standard)

    with pytest.raises(TimeoutError):
        asyncio.run(handler._generate_response("Hello", [], "", "another-model", 42, 0))

    mcp.assert_awaited_once()
    standard.assert_not_called()


def test_gemini_deadline_cancels_running_tool_without_followup(chat, monkeypatch):
    handler, sdk = chat
    sdk.return_value = gemini_response(
        types.Part(function_call=types.FunctionCall(name="lookup", args={}))
    )
    tools = [{"function": {"name": "lookup", "description": "Look up a value"}}]
    monkeypatch.setattr(
        conversation, "get_mcp_tools_for_user", AsyncMock(return_value=tools)
    )
    tool_cancelled = []

    async def stalled_tool(function_call):
        try:
            await asyncio.Event().wait()
        finally:
            tool_cancelled.append(function_call.name)

    monkeypatch.setattr(handler.gemini_api, "_execute_mcp_tool", stalled_tool)

    with pytest.raises(TimeoutError):
        generate(handler, timeout=0)

    assert tool_cancelled == ["lookup"]
    sdk.assert_awaited_once()


@pytest.mark.parametrize(
    ("use_fallback", "indicator"),
    [(False, "✨ Gemini 4 Hydrogen"), (True, "✨ Gemini 3.5 Flash-Lite")],
)
def test_telegram_labels_serving_model_and_saves_under_existing_model_key(
    chat, use_fallback, indicator
):
    handler, sdk = chat
    response = gemini_response(types.Part.from_text(text="Hi"))
    sdk.side_effect = [unavailable(), response] if use_fallback else [response]
    handler.prompt_formatter = SimpleNamespace(
        apply_response_guidelines=AsyncMock(return_value="Hello")
    )
    handler.conversation_manager = SimpleNamespace(
        get_intelligent_context=AsyncMock(return_value={})
    )
    handler._clean_response_content = str.strip
    handler._send_formatted_response = AsyncMock()
    handler._save_conversation = AsyncMock()
    handler.logger = logging.getLogger(__name__)
    update, context = SimpleNamespace(message=object()), object()
    placeholder = SimpleNamespace(delete=AsyncMock())

    asyncio.run(
        handler._handle_text_conversation(
            update,
            context,
            placeholder,
            "Hello",
            "",
            None,
            [],
            42,
            GEMINI_CHAT_ALIAS,
        )
    )

    placeholder.delete.assert_awaited_once()
    handler._send_formatted_response.assert_awaited_once_with(
        update,
        context,
        update.message,
        "Hi",
        indicator,
        "",
        None,
    )
    handler._save_conversation.assert_awaited_once_with(
        42, "Hello", "Hi", "", GEMINI_CHAT_ALIAS
    )
