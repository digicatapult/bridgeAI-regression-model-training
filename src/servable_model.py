"""Servable model class.

Custom MLFlow class is bundled with with preprocessing and dependancies."""

import joblib
import mlflow
import torch


class ServableModel(mlflow.pyfunc.PythonModel):
    def __init__(self, device):
        self.device = device

    def load_context(self, context):
        # Load the PyTorch model
        model_path = context.artifacts["torch_model"]
        model = torch.load(model_path, weights_only=False)
        self.model = model
        if self.device:
            self.model.to(self.device)
        self.model.eval()

        # Load the preprocessor
        preprocessor_path = context.artifacts["preprocessor"]
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, context, model_input):
        # Apply the preprocessor to the input data
        preprocessed_data = self.preprocessor.transform(model_input)

        # Convert the preprocessed data to a tensor
        input_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)

        # Run the model inference
        with torch.no_grad():
            output = self.model(input_tensor)

        return output.numpy()
