# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12-slim

# Install dependencies and Poetry
RUN apt-get update &&  \
    apt-get install -y --fix-missing build-essential git && \
    pip install --no-cache-dir poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Configure Poetry to not use virtual environments
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock ./

# Install only non-development dependencies
RUN poetry install --no-dev --no-root

# Copy the rest of the application code into the container
COPY config.yaml .
COPY src ./src
RUN mkdir artefacts

# Add source directory to python path
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

# Set the environment variable
# raw data path
ENV DATA_PATH=/app/artefacts/HousingData.csv
# config path
ENV CONFIG_PATH=./config.yaml
# log level
ENV LOG_LEVEL=INFO
# MLFlow tracking uri
ENV MLFLOW_TRACKING_URI="http://localhost:5000"

# Run the application
CMD ["poetry", "run", "python", "src/main.py"]
