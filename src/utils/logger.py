import logging
import logging.handlers
import os
import sys
from typing import Literal, Optional

from utils.colors import ANSIColors


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SystemLogFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'className'):
            record.className = '_'
        return True


class CustomFormatter(logging.Formatter):
    grey = ANSIColors.GRAY
    yellow = ANSIColors.YELLOW
    red = ANSIColors.RED
    bold_red = ANSIColors.BOLD_RED
    white = ANSIColors.WHITE
    reset = ANSIColors.RESET
    log_format = '%(asctime)s: %(levelname)s: [%(className)s.%(funcName)s]: %(message)s'

    FORMATS = {
        logging.DEBUG: grey + log_format + reset,
        logging.INFO: white + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger(metaclass=Singleton):
    """Singleton Logger
    """
    def __init__(
        self,
        save_path: str | os.PathLike[str] = '',
        filename: str = 'api.log',
        level: Optional[str] = 'info',
        mode: Literal['all', 'file', 'console'] = 'console',
        logger_name: str = 'default',
    ):
        self.save_path = save_path
        self.filename = filename
        self.level = level
        self.mode = mode
        self.logger_name = logger_name

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        self.logger = self.initialize()

    @classmethod
    def instance(cls) -> logging.Logger:
        return cls().logger

    def initialize(self) -> logging.Logger:
        logger = logging.getLogger(self.logger_name)
        logger.addFilter(SystemLogFilter())
        if (logger.hasHandlers()):
            logger.handlers.clear()

        if self.level and self.level.upper() in ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
            level = self.level.upper()
            logger.setLevel(getattr(logging, level))

        if self.mode == 'file' or self.mode == 'all':
            fileHandler = logging.handlers.RotatingFileHandler(
                os.path.join(self.save_path, self.filename),
                mode='a',
                maxBytes=50 * 1024 * 1024,
                backupCount=2,
                encoding=None,
            )
            fileHandler.setFormatter(CustomFormatter())
            logger.addHandler(fileHandler)

        if self.mode == 'console' or self.mode == 'all':
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(CustomFormatter())
            logger.addHandler(consoleHandler)

        return logger
