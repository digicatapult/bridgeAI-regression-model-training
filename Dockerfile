FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry==1.8.3

COPY poetry.lock pyproject.toml /app/

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-ansi