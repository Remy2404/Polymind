import asyncio

import httpx
from src.api.routes.webapp import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

def test_get_sessions_no_auth():
    # When no auth header or user_id param provided, the endpoint should return 401
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.get("/webapp/sessions")

    response = asyncio.run(request())
    assert response.status_code == 401
