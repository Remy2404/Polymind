# wsgi.py
from main import create_app, TelegramBot

telegram_bot = TelegramBot()
flask_app = create_app(telegram_bot)

app = flask_app