# CHANGELOG

## v0.3.0 (2025/12/07)
- feat:
	- Added simulator functionality to test trading strategies using historical data without risking real money.
	- Included a sample data file for the simulator at `sorul_tradingbot/strategy/simulator/data/sp500_5m_test.csv`.
	- Provided a debug configuration in `.vscode/launch.json` for easy execution of the simulator.
- fix:
	- Update urllib3 to 2.6.0 version to address security vulnerability.

## v0.2.2 (2025/07/29)
- fix:
	- Update urllib3 and requests dependencies to 2.5.0 and 2.32.4 versions respectively to address security vulnerabilities.

## v0.2.1 (2025/05/20)
- fix:
	- Update setuptools dev dependecy to 80.8.0 version for fixing vulnerability issue.

## V0.2.0 (2024/11/23)
- refactor:
  - Metatrader running in docker

## V0.1.1
- refactor:
  - Updating tradingbot to tradeo package.

## V0.1.0
- feat:
  - TNT private strategy.
  - A main executable for forex bot.