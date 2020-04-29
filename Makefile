venv: venv/bin/activate

venv/bin/activate: requirements.txt
	python3 -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate; pylint crpm ; pytest

clean:
	rm -rf venv
	rm -rf src
	find . -depth -name "*.pyc" -type f -delete
