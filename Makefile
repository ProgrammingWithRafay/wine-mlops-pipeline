.PHONY: install lint test train evaluate clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	python -m flake8 src tests --max-line-length=100

test:
	python -m pytest -v

train:
	python -m src.train

evaluate:
	python -m src.evaluate

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache') if '.venv' not in p.parts]; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc') if '.venv' not in p.parts]"
