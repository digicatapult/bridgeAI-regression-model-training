"""Utility functions."""

import logging

import yaml
from pythonjsonlogger import jsonlogger


def load_yaml_config(config_path: str = "./config.yaml"):
    """Load the json configuration."""
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom log formatter."""

    def add_fields(self, log_record, record, message_dict):
        """Adding statndard fileds for logging.

        #TODO: remove unwanted fields and add if anything needed
        """
        super(CustomJsonFormatter, self).add_fields(
            log_record, record, message_dict
        )
        # log_record['thread'] = record.thread
        # log_record['process'] = record.process
        log_record["module"] = record.module
        log_record["funcName"] = record.funcName
        log_record["pathname"] = record.pathname
        log_record["lineno"] = record.lineno
        log_record["filename"] = record.filename
        log_record["levelname"] = record.levelname


def setup_logger(log_file="./artefacts/app.log.json"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler to log to a file
    file_handler = logging.FileHandler(log_file)
    formatter = CustomJsonFormatter("%(asctime)s %(message)s")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


# Initialized logger
logger = setup_logger()
