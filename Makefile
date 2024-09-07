flake8:
	@flake8 --config config/tox.ini

test:
	@poetry run pytest

requirements:
	@poetry export -f requirements.txt --output requirements.txt --without-hashes

dev_requirements:
	@poetry export --dev -f requirements.txt --output requirements_dev.txt --without-hashes

master:
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

run_container:
	@sh ../MetaTrader5-Docker-Image/stop.sh
	@docker build -t sorul_tradingbot .
	@docker run --name sorul_tradingbot -d -p 3000:3000 -v ./metatrader:/config sorul_tradingbot