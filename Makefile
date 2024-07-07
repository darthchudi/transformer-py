freeze-pip:
	pip freeze > requirements.txt

install-from-requirements:
	pip install -r requirements.txt

run:
	@python3 main.py

lint:
	@ruff check