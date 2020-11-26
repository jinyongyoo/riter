import torch
import logging
import logging.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STRING = "\033[34;1mriter\033[0m"
logger = logging.getLogger(__name__)
logging.config.dictConfig(
    {"version": 1, "loggers": {__name__: {"level": logging.INFO}}}
)
formatter = logging.Formatter(f"{LOG_STRING}: %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.propagate = False


def type_check(type1, type2):
    if isinstance(type2, (tuple, list)):
        return type1 in type2
    else:
        return type1 == type2
