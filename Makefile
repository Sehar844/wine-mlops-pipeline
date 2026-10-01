.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v

train:
	python -m src.train

clean:
	del /S /Q *.pyc 2>nul || exit 0
	if exist .pytest_cache rmdir /S /Q .pytest_cache
	if exist .flake8_cache rmdir /S /Q .flake8_cache