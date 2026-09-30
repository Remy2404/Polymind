from fastapi import HTTPException, Request

def get_services(request: Request) -> dict:
    bot = getattr(request.app.state, "bot", None)
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot instance not initialized")

    text_handler = getattr(bot, "text_handler", None)
    memory_manager = getattr(text_handler, "memory_manager", None)
    conversation_manager = getattr(bot, "conversation_manager", None)
    user_data_manager = getattr(bot, "user_data_manager", None)
    if any(
        service is None
        for service in (memory_manager, conversation_manager, user_data_manager)
    ):
        raise HTTPException(status_code=503, detail="Bot services not initialized")

    return {
        "user_data_manager": user_data_manager,
        "memory_manager": memory_manager,
        "conversation_manager": conversation_manager,
        "gemini_api": getattr(bot, "gemini_api", None),
        "openrouter_api": getattr(bot, "openrouter_api", None),
        "deepseek_api": getattr(bot, "deepseek_api", None),
    }


def get_sessions_collection(request: Request):
    bot = getattr(request.app.state, "bot", None)
    db = getattr(bot, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db["chat_sessions"]
