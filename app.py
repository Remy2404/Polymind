import sys
import os
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from src.api.app_factory import create_application
import uvicorn

# Aggressive garbage collection for memory optimization
import gc
gc.set_threshold(50, 5, 5)
gc.collect()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# Single thread pool for low-resource instances
thread_pool = ThreadPoolExecutor(max_workers=1)

app = create_application()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
