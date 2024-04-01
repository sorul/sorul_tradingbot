"""Script."""
from datetime import datetime
from tradingbot.config import Config


def run() -> None:
  print(datetime.now(Config.local_timezone))


if __name__ == "__main__":
  run()
