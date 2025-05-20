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
	@git push
	@git checkout develop
	@poetry version

start_docker:
	docker compose up -d

stop_docker:
	docker compose down

run_forex:
	source .env && ~/.local/bin/poetry run run_forex