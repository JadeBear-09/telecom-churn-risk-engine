.PHONY: install train evaluate batch-score api dashboard test

install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

evaluate:
	python -m src.evaluate

batch-score:
	python -m src.batch_scoring

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/streamlit_app.py --server.port 8501

test:
	pytest -q
