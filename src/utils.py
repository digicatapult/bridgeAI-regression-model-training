"""Utility functions."""

import logging
import os
import sys

import yaml
from pythonjsonlogger import jsonlogger


def load_yaml_config(config_path: str):
    """Load the json configuration."""
    config_path = os.getenv("CONFIG_PATH", "./config.yaml")
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom log formatter."""

    def add_fields(self, log_record, record, message_dict):
        """Adding standard filed for logging."""
        super(CustomJsonFormatter, self).add_fields(
            log_record, record, message_dict
        )
        log_record["module"] = record.module
        log_record["funcName"] = record.funcName
        log_record["pathname"] = record.pathname
        log_record["lineno"] = record.lineno
        log_record["filename"] = record.filename
        log_record["levelname"] = record.levelname


def setup_logger():
    logger = logging.getLogger()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # set the logging level to info if the provided one is invalid
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Create a stream handler to log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter("%(asctime)s %(message)s")
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(stream_handler)

    return logger


# Initialized logger
logger = setup_logger()
