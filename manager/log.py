import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("manager_logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "manager.log"

def init_logger(level=logging.INFO):
    logger = logging.getLogger("manager")
    logger.setLevel(level)

    if logger.handlers:
        return logger
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)

    # Rotating File Handler
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes = 5 * 1024 * 1024,
        backupCount=5,
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)"
    ))
    logger.addHandler(fh)

    return logger

LOGGER = init_logger()

def dbg(msg: str):
    LOGGER.debug(msg)

def info(msg: str):
    LOGGER.info(msg)

def warn(msg: str):
    LOGGER.warning(msg)

def err(msg: str):
    LOGGER.error(msg)

def get_network_logger(network_id: str):
    logger = logging.getLogger(f"manager.{network_id}")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    net_dir = LOG_DIR / network_id
    net_dir.mkdir(exist_ok=True)

    log_path = net_dir/ "manager.log"
    fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)"
    ))
    logger.addHandler(fh)

    return logger
