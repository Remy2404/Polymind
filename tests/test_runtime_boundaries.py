import asyncio
import importlib
import logging
import pytest
import httpx

from fastapi import FastAPI
from fastapi import HTTPException
from starlette.requests import Request
from types import SimpleNamespace

from src.api.app_factory import create_application, lifespan_context
from src.api.dependencies import get_services, get_sessions_collection
from src.api.routes import webapp, webapp_streaming, webhook
from src.api.webapp import legacy, message_routes, session_routes
from src.ai.models.factory import ModelHandlerFactory
from src.ai.models.model_configs import ModelConfig, ModelConfigurations, Provider
from src.ai.providers.gemini_api import GeminiAPI
from src.ai.providers.openrouter_api import OpenRouterAPI
from src.bot.telegram_bot import TelegramBot


def test_model_routing_uses_the_providers_passed_for_each_application(monkeypatch):
    config = ModelConfig(model_id="gemini", display_name="Gemini", provider=Provider.GEMINI)
    monkeypatch.setattr(ModelConfigurations, "get_all_models", lambda: {"gemini": config})
    first_api = object()
    second_api = object()

    assert ModelHandlerFactory.get_model_handler("gemini", gemini_api=first_api) is first_api
    assert ModelHandlerFactory.get_model_handler("gemini", gemini_api=second_api) is second_api


def test_provider_credentials_are_resolved_at_construction(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiAPI(rate_limiter=object())
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterAPI(rate_limiter=object())


def test_media_processor_imports_from_the_ai_provider_package():
    module = importlib.import_module("src.media.processors.multi_file_processor")
    assert module.MultiFileProcessor.__name__ == "MultiFileProcessor"


def test_mcp_boundaries_preserve_configuration_and_tool_conversion(monkeypatch):
    from src.integrations.mcp import MCPManager, MCPServerClient, MCPToolConverter

    config = importlib.import_module("src.integrations.mcp.config")
    converter = importlib.import_module("src.integrations.mcp.tool_converter")
    server = importlib.import_module("src.integrations.mcp.server_client")
    manager = importlib.import_module("src.integrations.mcp.manager")
    monkeypatch.setenv("MCP_TEST_KEY", "resolved")

    assert config.substitute_env_vars({"env": {"key": "$MCP_TEST_KEY"}}) == {
        "env": {"key": "resolved"}
    }
    tool = SimpleNamespace(
        name="lookup",
        description="Look up a value",
        inputSchema={"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]},
        outputSchema=None,
    )
    converted = converter.MCPToolConverter.convert_mcp_tool_to_openai(tool)
    assert converted["function"]["name"] == "lookup"
    assert converted["function"]["parameters"]["required"] == ["term"]
    assert MCPToolConverter is converter.MCPToolConverter
    assert MCPServerClient is server.MCPServerClient
    assert MCPManager is manager.MCPManager


def test_polling_lifecycle_stays_on_the_application_event_loop(monkeypatch):
    events = []

    async def initialize_mcp():
        events.append("mcp")

    async def cleanup_mcp():
        events.append("mcp_shutdown")

    async def record(name):
        events.append(name)

    class Updater:
        async def start_polling(self):
            await record("polling_start")

        async def stop(self):
            await record("polling_stop")

    class Application:
        updater = Updater()

        async def initialize(self):
            await record("initialize")

        async def start(self):
            await record("start")

        async def stop(self):
            await record("stop")

        async def shutdown(self):
            await record("shutdown")

        def run_polling(self):
            events.append("run_polling")

    class Bot:
        application = Application()
        logger = logging.getLogger(__name__)
        client = None

        async def shutdown(self):
            await record("bot_shutdown")

    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.setattr("src.api.app_factory.initialize_mcp_for_bot", initialize_mcp)
    monkeypatch.setattr("src.api.app_factory.mcp_integration.cleanup", cleanup_mcp)

    async def run():
        async with lifespan_context(FastAPI(), Bot()):
            events.append("serving")

    asyncio.run(run())
    assert events == [
        "mcp", "initialize", "start", "polling_start", "serving",
        "polling_stop", "stop", "shutdown", "bot_shutdown", "mcp_shutdown",
    ]


def test_shutdown_closes_remaining_resources_after_stop_failure(monkeypatch):
    events = []

    async def initialize_mcp():
        return None

    async def cleanup_mcp():
        events.append("mcp_shutdown")

    class Application:
        async def initialize(self):
            events.append("initialize")

        async def start(self):
            events.append("start")

        async def stop(self):
            events.append("stop")
            raise RuntimeError("stop failed")

        async def shutdown(self):
            events.append("shutdown")

    class Bot:
        application = Application()

        async def setup_webhook(self):
            events.append("webhook")

        async def shutdown(self):
            events.append("bot_shutdown")

    monkeypatch.setenv("WEBHOOK_URL", "https://example.test")
    monkeypatch.setattr("src.api.app_factory.initialize_mcp_for_bot", initialize_mcp)
    monkeypatch.setattr("src.api.app_factory.mcp_integration.cleanup", cleanup_mcp)

    async def run():
        async with lifespan_context(FastAPI(), Bot()):
            events.append("serving")

    with pytest.raises(RuntimeError, match="stop failed"):
        asyncio.run(run())
    assert events == [
        "initialize", "start", "webhook", "serving", "stop", "shutdown",
        "bot_shutdown", "mcp_shutdown",
    ]


def test_webhook_bot_is_scoped_to_the_request_application():
    first = FastAPI()
    second = FastAPI()
    first.state.bot = object()
    second.state.bot = object()

    first_request = Request({"type": "http", "app": first})
    second_request = Request({"type": "http", "app": second})
    assert webhook.get_telegram_bot(first_request) is first.state.bot
    assert webhook.get_telegram_bot(second_request) is second.state.bot


def test_api_dependencies_reuse_the_bot_owned_services_and_database():
    memory = object()
    conversation = object()
    collection = object()
    bot = SimpleNamespace(
        db={"chat_sessions": collection},
        user_data_manager=object(),
        text_handler=SimpleNamespace(memory_manager=memory),
        conversation_manager=conversation,
        gemini_api=object(),
        openrouter_api=object(),
        deepseek_api=object(),
    )
    app = FastAPI()
    app.state.bot = bot
    request = Request({"type": "http", "app": app})

    services = get_services(request)
    assert services["memory_manager"] is memory
    assert services["conversation_manager"] is conversation
    assert services["user_data_manager"] is bot.user_data_manager
    assert services["gemini_api"] is bot.gemini_api
    assert get_sessions_collection(request) is collection


def test_legacy_session_mapping_cache_is_scoped_to_user_and_application(monkeypatch):
    async def build_mapping(user, request):
        return {str(user.id): str(id(request.app))}

    monkeypatch.setattr(legacy, "_build_session_mapping", build_mapping)
    first_app = FastAPI()
    second_app = FastAPI()
    first_request = Request({"type": "http", "app": first_app})
    second_request = Request({"type": "http", "app": second_app})
    first_user = SimpleNamespace(id=1)
    second_user = SimpleNamespace(id=2)

    async def run():
        assert await legacy.get_cached_session_mapping(first_user, first_request) == {
            "1": str(id(first_app))
        }
        assert await legacy.get_cached_session_mapping(second_user, first_request) == {
            "2": str(id(first_app))
        }
        assert await legacy.get_cached_session_mapping(first_user, second_request) == {
            "1": str(id(second_app))
        }

    asyncio.run(run())


def test_legacy_session_endpoint_does_not_append_old_sessions_twice(monkeypatch):
    new_session = SimpleNamespace(updated_at="2026-01-02")
    old_session = SimpleNamespace(updated_at="2026-01-01")

    async def list_sessions(*args, **kwargs):
        return [new_session, old_session]

    async def list_old_sessions(*args, **kwargs):
        return [old_session]

    monkeypatch.setattr(session_routes, "get_chat_sessions", list_sessions)
    monkeypatch.setattr(session_routes, "get_old_format_sessions", list_old_sessions)
    request = Request({"type": "http", "app": FastAPI()})
    result = asyncio.run(
        session_routes.get_chat_sessions_legacy(
            request, SimpleNamespace(id=1), False, limit=50, offset=0
        )
    )
    assert result == [new_session, old_session]


def test_legacy_chat_listing_does_not_apply_offset_twice(monkeypatch):
    session = SimpleNamespace(
        id="second", title="Second", model="gemini", created_at="2026-01-01",
        updated_at="2026-01-02", message_count=0, user_id="1", pinned=False,
        pinned_at=None,
    )

    async def paged_sessions(*args, **kwargs):
        return [session]

    monkeypatch.setattr(session_routes, "get_chat_sessions", paged_sessions)
    request = Request({"type": "http", "app": FastAPI()})
    result = asyncio.run(
        session_routes.list_chats_legacy(request, SimpleNamespace(id=1), limit=1, offset=1)
    )
    assert [chat["id"] for chat in result] == ["second"]


def test_legacy_chat_delete_uses_the_current_application_database():
    deleted = []

    class Collection:
        def __init__(self, count):
            self.count = count

        def delete_one(self, query):
            deleted.append(query)
            return SimpleNamespace(deleted_count=self.count)

    app = FastAPI()
    app.state.bot = SimpleNamespace(
        db={"chat_sessions": Collection(0), "conversations": Collection(1)}
    )
    request = Request({"type": "http", "app": app})
    result = asyncio.run(
        session_routes.delete_chat_legacy(
            request, "user_1_model_gemini", SimpleNamespace(id=1)
        )
    )

    assert result["status"] == "success"
    assert deleted == [
        {"session_id": "user_1_model_gemini", "user_id": "1"},
        {"cache_key": "user_1_model_gemini"},
    ]


def test_application_can_be_composed_without_external_services():
    bot = SimpleNamespace(token="test-token")
    app = create_application(bot=bot)
    app.openapi()

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.get("/webapp/health")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["bot_initialized"] is True


def test_webapp_routes_preserve_the_public_paths_and_methods():
    assert {(method, route.path) for route in webapp.router.routes for method in route.methods} == {
        ("GET", "/webapp/user"),
        ("GET", "/webapp/user/preferences"),
        ("POST", "/webapp/user/preferences"),
        ("POST", "/webapp/v2/sessions"),
        ("GET", "/webapp/v2/sessions"),
        ("GET", "/webapp/sessions"),
        ("GET", "/webapp/v2/sessions/{session_id}/messages"),
        ("GET", "/webapp/messages/{chat_id:path}"),
        ("POST", "/webapp/chat"),
        ("GET", "/webapp/models"),
        ("GET", "/webapp/health"),
        ("POST", "/webapp/chats"),
        ("GET", "/webapp/chats"),
        ("GET", "/webapp/history"),
        ("DELETE", "/webapp/chats/{chat_id:path}"),
    }


def test_streaming_router_only_adds_streaming_specific_routes():
    regular_routes = {
        (method, route.path)
        for route in webapp.router.routes
        for method in route.methods
    }
    streaming_routes = {
        (method, route.path)
        for route in webapp_streaming.router.routes
        for method in route.methods
    }
    assert regular_routes.isdisjoint(streaming_routes)
    assert streaming_routes == {
        ("POST", "/webapp/chat/stream"),
        ("GET", "/webapp/chats/{chat_id}/messages"),
    }


def test_streaming_chat_messages_use_app_database_and_enforce_ownership(monkeypatch):
    documents = {
        "mine": {"session_id": "mine", "user_id": "1", "messages": [{"content": "hello"}]},
        "other": {"session_id": "other", "user_id": "2", "messages": [{"content": "private"}]},
    }

    class Collection:
        def find_one(self, query):
            return next(
                (doc for doc in documents.values() if all(doc.get(key) == value for key, value in query.items())),
                None,
            )

    monkeypatch.setattr(webapp_streaming, "get_sessions_collection", lambda request: Collection(), raising=False)
    request = Request({"type": "http", "app": FastAPI()})
    user = SimpleNamespace(id=1)

    result = asyncio.run(webapp_streaming.get_chat_messages(request, "mine", user, 100, 0))
    assert result == {"messages": [{"content": "hello"}], "total": 1, "has_more": False}
    with pytest.raises(HTTPException) as error:
        asyncio.run(webapp_streaming.get_chat_messages(request, "other", user, 100, 0))
    assert error.value.status_code == 404


def test_legacy_message_route_passes_request_to_uuid_lookup(monkeypatch):
    request = Request({"type": "http", "app": FastAPI()})
    user = SimpleNamespace(id=1)

    async def mapping(current_user, current_request):
        return {}

    async def messages(current_request, chat_id, current_user):
        assert current_request is request
        assert chat_id == "session-uuid"
        assert current_user is user
        return ["stored message"]

    monkeypatch.setattr(message_routes, "get_cached_session_mapping", mapping)
    monkeypatch.setattr(message_routes, "get_session_messages", messages)
    result = asyncio.run(message_routes.get_messages_legacy(request, "session-uuid", None, user))
    assert result == ["stored message"]


def test_legacy_message_formats_keep_the_same_public_fields(monkeypatch):
    async def mapping(current_user, request):
        return {}

    async def history(user_id, model, max_messages):
        return [
            {"role": "assistant", "content": "reply", "timestamp": "2026-01-01"},
            {"content": "legacy", "timestamp": "2026-01-02"},
        ]

    monkeypatch.setattr(message_routes, "get_cached_session_mapping", mapping)
    monkeypatch.setattr(
        message_routes,
        "get_services",
        lambda request: {
            "conversation_manager": SimpleNamespace(get_conversation_history=history)
        },
    )
    request = Request({"type": "http", "app": FastAPI()})
    messages = asyncio.run(
        message_routes.get_messages_legacy(
            request, "user_1_model_gemini", None, SimpleNamespace(id=1)
        )
    )
    assert [(message.id, message.role, message.content, message.createdAt, message.model) for message in messages] == [
        ("msg_0", "assistant", "reply", "2026-01-01", "gemini"),
        ("msg_1", "user", "legacy", "2026-01-02", "gemini"),
    ]


def test_webhook_processing_propagates_shutdown_cancellation():
    class Bot:
        application = SimpleNamespace(bot=SimpleNamespace(username="test"))

        async def process_update(self, update):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            webhook._process_update_with_retry(
                Bot(), {"update_id": 1}, logging.getLogger(__name__)
            )
        )


def test_bot_resource_cleanup_continues_after_provider_failure(monkeypatch):
    events = []

    class Resource:
        closed = False

        async def close(self):
            events.append("session")

    class Gemini:
        async def close(self):
            events.append("gemini")
            raise RuntimeError("gemini close failed")

    class OpenRouter:
        async def close(self):
            events.append("openrouter")

    bot = TelegramBot.__new__(TelegramBot)
    bot.session = Resource()
    bot.gemini_api = Gemini()
    bot.openrouter_api = OpenRouter()
    bot.client = object()
    monkeypatch.setattr(
        "src.bot.telegram_bot.close_database_connection",
        lambda client: events.append("database"),
    )

    with pytest.raises(RuntimeError, match="gemini close failed"):
        asyncio.run(bot.shutdown())
    assert events == ["session", "gemini", "openrouter", "database"]
