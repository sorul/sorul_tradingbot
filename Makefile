SHELL := /bin/bash

flake8:
	poetry run flake8 --config config/tox.ini

test:
	poetry run pytest

requirements:
	poetry lock
	poetry export -f requirements.txt --output requirements.txt --without-hashes

dev_requirements:
	poetry lock
	poetry export --with dev -f requirements.txt --output requirements_dev.txt --without-hashes

tag:
	@make flake8
	@make requirements
	@make dev_requirements
	@git add .
	@git commit -am "v$$(poetry version -s)"
	@git push
	@git checkout master
	@git merge --no-edit --log develop
	@git tag v$$(poetry version -s)
	@git push --tags
	@git checkout develop
	@poetry version
	@echo "Tagging complete. Make a pull request to merge developer into master -> https://github.com/sorul/sorul_tradingbot/compare/develop?expand=1"

start_docker:
	docker compose up -d

stop_docker:
	docker compose down

run_forex:
	source .env && ~/.local/bin/poetry run run_forex

simulator_50k:
	nohup poetry run python sorul_tradingbot/strategy/simulator/simulator.py --data-file sorul_tradingbot/strategy/simulator/data/sp500_50k.csv --strategy-module sorul_tradingbot.strategy.private.volume_16 --start-date "" --finish-date "" > simulator.log 2>&1 &

simulator_2024:
	nohup poetry run python sorul_tradingbot/strategy/simulator/simulator.py --data-file sorul_tradingbot/strategy/simulator/data/sp500_2024.csv --strategy-module sorul_tradingbot.strategy.private.volume_16 --start-date "" --finish-date "" > simulator.log 2>&1 &

simulator_2023:
	nohup poetry run python sorul_tradingbot/strategy/simulator/simulator.py --data-file sorul_tradingbot/strategy/simulator/data/sp500_2023.csv --strategy-module sorul_tradingbot.strategy.private.volume_16 --start-date "" --finish-date "" > simulator.log 2>&1 &

simulator_2022:
	nohup poetry run python sorul_tradingbot/strategy/simulator/simulator.py --data-file sorul_tradingbot/strategy/simulator/data/sp500_2022.csv --strategy-module sorul_tradingbot.strategy.private.volume_16 --start-date "" --finish-date "" > simulator.log 2>&1 &