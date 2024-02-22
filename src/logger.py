import logging
import os
from os.path import dirname, join
import time
from datetime import timedelta

import pandas as pd

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s - %s" % (
            record.levelname,
            record.module,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""



def create_logger(module_name:str, filename:str=None)->logging.Logger:
    """
    Create a logger.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filename is None:
        filename = module_name

    path_log = str(os.path.join(ROOT_PATH, "log", filename))

    if not path_log.endswith(".log"):
        path_log += ".log"
    
    
    if not os.path.exists(dirname(path_log)):
        os.makedirs(dirname(path_log))

    file_handler = logging.FileHandler(path_log, "a")

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger(module_name)
    
    logger.handlers = []
    
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger

class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)


if __name__ == "__main__":
    from time import sleep
    logger = create_logger(__name__, filename="test_data_version")

    logger.info("Hi")

    logger = create_logger(__name__, filename="test_data_version.log")

    logger.info("Hi")

    logger = create_logger(__name__)

    logger.info("Hi")