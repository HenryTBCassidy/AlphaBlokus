import atexit
import json
import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Any, Dict

from dataclass_wizard import fromdict

from core.config import RunConfig


class AverageMeter:
    """
    Computes and stores the average and current value.
    Originally from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self) -> None:
        """Initialise the meter with zero values."""
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def __repr__(self) -> str:
        """Return string representation of the average value."""
        return f'{self.avg:.2e}'

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.

        Args:
            val: The new value to include in the average
            n: The weight of the new value (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_args(filename: str) -> RunConfig:
    """
    Load run configuration from a JSON file.

    Args:
        filename: Name of the JSON configuration file

    Returns:
        RunConfig: Configuration object for the run
    """
    root_dir = Path("run_configurations")
    with open(root_dir / filename, "r") as f:
        args_json = json.load(f)

    return fromdict(RunConfig, args_json)


def setup_logging(log_dir: Path) -> None:
    """
    Set up logging configuration from a JSON file.

    Args:
        log_dir: Directory where log files will be stored
    """
    config_file = Path("logging_config.json")
    with open(config_file) as f_in:
        config: Dict[str, Any] = json.load(f_in)

    log_dir.mkdir(exist_ok=True, parents=True)
    config["handlers"]["file_log"]["filename"] = f"{log_dir / 'alpha.log'}"
    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
