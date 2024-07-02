# bridgeAI-regression-model-training

### Model training;

1. The data used is available [here](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).
Download the csv file and update the path to the csv file in the `config.yaml` file in `data.raw_data` 
or in the environment variable `DATA_PATH`
2. Update the python environment in `.env` file
3. Install the dependancies using poetry `poetry install --no-dev`
4. update the config and model parameters in the `config.yaml` file
5. Add `./src` to the `PYTHONPATH` - `export PYTHONPATH="${PYTHONPATH}:./src"`
6. Run `python src/main.py` or `poetry run python src/main.py`


#### Using docker;
1. Build the docker image - `docker build -t regression .`
2. Run the container with the correct `DATA_PATH` as environment variable `docker run -e DATA_PATH=/app/artefacts/HousingData.csv -v ./artefacts:/app/artefacts --rm regression`