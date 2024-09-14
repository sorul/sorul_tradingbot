flake8:
	@poetry run flake8 --config config/tox.ini

test:
	@poetry run pytest

requirements:
	@poetry export -f requirements.txt --output requirements.txt --without-hashes

dev_requirements:
	@poetry export --dev -f requirements.txt --output requirements_dev.txt --without-hashes

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

run_docker:
	docker compose up -d

stop_docker:
	docker compose down