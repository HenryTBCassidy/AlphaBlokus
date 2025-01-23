import json
import logging
import pathlib

import atexit
import logging.config
import logging.handlers


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logging(log_dir: pathlib.Path):
    config_file = pathlib.Path("logging_config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)

    log_dir.mkdir(exist_ok=True, parents=True)
    config["handlers"]["file_log"]["filename"] = f"{log_dir / 'alpha.log'}"
    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
