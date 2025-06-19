from loguru import logger
import sys

LOG_DIR = "logs"
ROTATION_SIZE = "10 MB"
RETENTION = "3 days"

# logger.remove(0)
# logger.add(sys.stderr, format="{time:MMMM D, YYYY > HH:mm:ss} | {level} | {message} | {extra}")
logger.add(f"{LOG_DIR}/log.log", 
            level="INFO",  rotation=ROTATION_SIZE, compression="zip", 
            retention=RETENTION, serialize=True, 
            format="{time:MMMM D, YYYY > HH:mm:ss} | {level} | {message} | {extra}")

def get_logger():
    return logger