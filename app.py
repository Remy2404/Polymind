import sys
import logging
from dotenv import load_dotenv
from src.api.app_factory import create_application
import uvicorn

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
app = create_application()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
