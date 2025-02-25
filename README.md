# bridgeAI-regression-model-training

## Model training

1. The data used is available [here](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).
Download the csv file and update the path to the csv file in the `config.yaml` file in `data.raw_data` 
or in the environment variable `DATA_PATH`
2. Update the python environment in `.env` file
3. Install `poetry` if not already installed
4. Install the dependencies using poetry `poetry install`
5. update the config and model parameters in the `config.yaml` file
6. Add `./src` to the `PYTHONPATH` - `export PYTHONPATH="${PYTHONPATH}:./src"`
7. Run `python src/main.py` or `poetry run python src/main.py`


### Model training - using docker
1. Build the docker image - `docker build -t regression .`
2. Bring up the dependencies by using `docker compose up -d`
3. Run the container with the correct `DATA_PATH` and `MLFLOW_TRACKING_URI` as environment variables.
   (Refer to the following [Environment Variables](#environment-variables) table for complete list)\
   `docker run -e DATA_PATH=/app/artefacts/HousingData.csv -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 -v ./artefacts:/app/artefacts --rm regression`


### Environment Variables

The following environment variables can be set to configure the training:

| Variable              | Default Value                 | Description                                                                                                                                                            |
|-----------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DATA_PATH             | `./artefacts/HousingData.csv` | File path to the raw data CSV data used for training                                                                                                                   |
| CONFIG_PATH           | `./config.yaml`               | File path to the model training and other configuration file                                                                                                           |
| LOG_LEVEL             | `INFO`                        | The logging level for the application. Valid values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.                                                           |
| MLFLOW_TRACKING_URI   | `http://localhost:5000`       | MLFlow tracking URI. Use `http://host.docker.internal:5000` if the MLFlow is running within docker container.                                                          |
| GITHUB_USERNAME       | None                          | Githuib username. This is needed to pull the data form the dvc repo.                                                                                                   |
| GITHUB_PASSWORD       | None                          | Githuib token. This is needed to pull the data form the dvc repo.                                                                                                      |
| DVC_REMOTE            | `s3://artifacts`              | Dvc remote                                                                                                                                                             |
| DVC_REMOTE_NAME       | `regression-model-remote`     | Dvc remote name.                                                                                                                                                       |
| DVC_ENDPOINT_URL      | `http://minio`                | The URL endpoint for the DVC storage backend. This is typically the URL of an S3-compatible service, such as MinIO, used to store and manage datasets and model files. |
| AWS_DEFAULT_REGION    | `eu-west-2`                   | The dvc remote s3 bucket region                                                                                                                                        |
| DVC_ACCESS_KEY_ID     | None                          | Access key id for dvc remote. Optional. Not needed if using IAM based access for dvc remote.                                                                           |
| DVC_SECRET_ACCESS_KEY | None                          | secret access key for dvc remote. Optional. Not needed if using IAM based access for dvc remote.                                                                       |
| DEPLOY_AS_CODE        | False                         | Should manual intervention or evaluation is needed to register the trained model or not.                                                                               |
| DEPLOY_MODEL_NAME     | `house_price_prediction`      | The name with which the model will be registered. This name and alias will be used for deployment.                                                                     |
| DEPLOY_MODEL_ALIAS    | `champion`                    | The alias to be added to the model. This alias along with the name will be used for deployment.                                                                        |


### Running the tests

Ensure that you have the project requirements already set up by following the [Model training](#model-training) instructions
- Ensure `pytest` is installed. `poetry install` will install it as a dev dependency.
- - For integration tests, set up the dependencies (MLFlow) by running, `docker-compose up -d`
- Run the tests with `poetry run pytest ./tests`
