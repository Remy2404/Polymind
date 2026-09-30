import asyncio
from types import SimpleNamespace

import httpx

from src.api.app_factory import create_application
from src.api.auth import get_current_user
from src.api.schemas.webapp import UserInfo
from src.api.webapp import message_routes


class SessionCollection:
    def __init__(self):
        self.documents = {}
        self.inserts = 0
        self.updates = 0

    def find_one(self, query):
        return self.documents.get(query["session_id"])

    def insert_one(self, document):
        self.documents[document["session_id"]] = document
        self.inserts += 1

    def update_one(self, query, update):
        document = self.documents[query["session_id"]]
        document["messages"].extend(update["$push"]["messages"]["$each"])
        document["updated_at"] = update["$set"]["updated_at"]
        self.updates += 1


class GeminiProvider:
    def __init__(self):
        self.prompts = []

    async def generate_response(self, prompt, model):
        self.prompts.append((prompt, model))
        return "A verified reply"


def test_chat_request_routes_to_provider_and_persists_one_message_pair(monkeypatch):
    sessions = SessionCollection()
    provider = GeminiProvider()
    bot = SimpleNamespace(token="test-token")
    app = create_application(bot=bot)
    app.dependency_overrides[get_current_user] = lambda: UserInfo(id=7, first_name="Ada")
    monkeypatch.setattr(message_routes, "get_sessions_collection", lambda request: sessions)
    monkeypatch.setattr(
        message_routes,
        "get_services",
        lambda request: {"gemini_api": provider, "openrouter_api": None, "deepseek_api": None},
    )

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.post(
                "/webapp/chat", json={"message": "Hello", "model": "gemini"}
            )

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["response"] == "A verified reply"
    assert provider.prompts == [("Hello", "gemini")]
    assert sessions.inserts == 1
    assert sessions.updates == 0
    session = sessions.documents[response.json()["session_id"]]
    assert session["user_id"] == "7"
    assert [message["role"] for message in session["messages"]] == ["user", "assistant"]


def test_chat_rejects_another_users_session_without_writing(monkeypatch):
    sessions = SessionCollection()
    sessions.documents["private-session"] = {
        "session_id": "private-session", "user_id": "99", "messages": []
    }
    provider = GeminiProvider()
    app = create_application(bot=SimpleNamespace(token="test-token"))
    app.dependency_overrides[get_current_user] = lambda: UserInfo(id=7, first_name="Ada")
    monkeypatch.setattr(message_routes, "get_sessions_collection", lambda request: sessions)
    monkeypatch.setattr(
        message_routes,
        "get_services",
        lambda request: {"gemini_api": provider, "openrouter_api": None, "deepseek_api": None},
    )

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.post(
                "/webapp/chat",
                json={"message": "Hello", "model": "gemini", "session_id": "private-session"},
            )

    response = asyncio.run(request())
    assert response.status_code == 403
    assert sessions.inserts == 0
    assert sessions.updates == 0
    assert provider.prompts == []


def test_chat_appends_one_message_pair_to_an_existing_owned_session(monkeypatch):
    sessions = SessionCollection()
    sessions.documents["existing"] = {
        "session_id": "existing", "user_id": "7", "messages": []
    }
    provider = GeminiProvider()
    app = create_application(bot=SimpleNamespace(token="test-token"))
    app.dependency_overrides[get_current_user] = lambda: UserInfo(id=7, first_name="Ada")
    monkeypatch.setattr(message_routes, "get_sessions_collection", lambda request: sessions)
    monkeypatch.setattr(
        message_routes,
        "get_services",
        lambda request: {"gemini_api": provider, "openrouter_api": None, "deepseek_api": None},
    )

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.post(
                "/webapp/chat",
                json={"message": "Hello", "model": "gemini", "session_id": "existing"},
            )

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["session_id"] == "existing"
    assert sessions.inserts == 0
    assert sessions.updates == 1
    assert [message["role"] for message in sessions.documents["existing"]["messages"]] == [
        "user", "assistant"
    ]


def test_chat_provider_failure_does_not_expose_internal_details(monkeypatch):
    class FailingProvider:
        async def generate_response(self, prompt, model):
            raise RuntimeError("private provider credential")

    sessions = SessionCollection()
    app = create_application(bot=SimpleNamespace(token="test-token"))
    app.dependency_overrides[get_current_user] = lambda: UserInfo(id=7, first_name="Ada")
    monkeypatch.setattr(message_routes, "get_sessions_collection", lambda request: sessions)
    monkeypatch.setattr(
        message_routes,
        "get_services",
        lambda request: {
            "gemini_api": FailingProvider(),
            "openrouter_api": None,
            "deepseek_api": None,
        },
    )

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.post(
                "/webapp/chat", json={"message": "Hello", "model": "gemini"}
            )

    response = asyncio.run(request())
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to generate response"}
    assert sessions.inserts == 0
    assert sessions.updates == 0
