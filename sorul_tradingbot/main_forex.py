"""Main Forex Script."""
from sorul_tradingbot.executable.forex_executable import ForexExecutable


def run() -> None:
  """Run the executable object."""
  ForexExecutable().entry_point()


if __name__ == '__main__':
  run()
