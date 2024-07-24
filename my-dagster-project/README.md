## Orchestrating ML Workflows with Dagster

### Introduction

Dagster is an orchestrator that's designed for developing and maintaining data assets, such as tables, data sets, machine learning models, and reports.

### Initial Setup

1. Install Pip - a package installer for Python https://pip.pypa.io/en/stable/installation/
2. Check your python version - `python --version`: Dagster supports `Python 3.8` through `3.12`.
3. Setup a virtual environment - `python -m venv .venv`

   - `-m` flag is used to run a module as a script
   - `venv` is the module that creates virtual environments
   - `.venv` is the name of the virtual environment

4. Activate the virtual environment - `source .venv/bin/activate`
   - `source` is a shell built-in command that executes the content of the file passed as argument in the current shell
   - `.venv/bin/activate` is the path to the activate script in the virtual environment
5. Install Dagster with a default project skeleton - `pip install dagster
dagster project scaffold --name my-dagster-project
`
6. Change directory to the project - `cd my-dagster-project`
7. Install dependencies - `pip install -e ".[dev]"`
   - `-e` flag is used to install a project in editable mode
   - `".[dev]"` installs the development dependencies
8. Run Dagster - `dagster dev`
9. Open the Dagster instance in your browser - `http://127.0.0.1:3000/locations`

### Creating your first asset in Dagster

1. Open `assets.py` under `my_dagster_project` directory
2. Define a new asset

   ```python
   from dagster import asset


   @asset
   def hello_world():
       message = "Hello, World!"
       print(message)

       return message
   ```

3. Open Dagster UI in the browser and navigate to the `Assets` tab -> `http://127.0.0.1:3000/assets`
4. Click on the `hello_world` asset to view the asset details
5. Click `View global asset lineage`
6. Click `Materialize` in the upper right side
   - Materialization refers to the process of creating and producing the output from a particular asset
7. Click `View` to view the materialized output in the `Runs` tab

### Setup a Linear Regression model asset graph

1. Install `scikit-learn` - `pip install scikit-learn`
   1. Cancel dagster run
   2. Add `scikit-learn` to the `setup.py`
   3. Update installed package with `pip install -e ".[dev]"`
2. Open `assets.py` under `my_dagster_project` directory
3. Define `load_data` asset

   ```python
    from sklearn import datasets
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split

    from dagster import asset, job

   @asset
   def load_data():
      """
      Load the Iris dataset.

      Returns:
        tuple: A tuple containing the features (X) and the target (y) of the Iris dataset.
      """
      X, y = datasets.load_iris(return_X_y=True)  # Changed to load_iris
      return X, y
   ```

4. Define `split_data` asset

   ```python
   @asset
   def split_data(load_data):
       """
       Split the Iris dataset into training and testing sets.

       Args:
           load_data (tuple): A tuple containing the features (X) and the target (y) of the Iris dataset.

       Returns:
           tuple: A tuple containing the training and testing sets for features (X_train, X_test) and target (y_train, y_test).
       """
       X, y = load_data
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=42
       )
       return X_train, X_test, y_train, y_test
   ```

5. Define the `train_model` asset

   ```python
   @asset
   def train_model(split_data):
       """
       Train a Linear Regression model on the training data.

       Args:
           split_data (tuple): A tuple containing the training and testing sets for features (X_train, X_test) and target (y_train, y_test).

       Returns:
           tuple: A tuple containing the trained model, parameters, mean squared error, R^2 score, training features, and training target.
       """
       X_train, X_test, y_train, y_test = split_data
       params = {
           "fit_intercept": True,
       }
       lr = LinearRegression(**params)
       lr.fit(X_train, y_train)
       y_pred = lr.predict(X_test)
       # Calculate metrics
       mse = mean_squared_error(y_test, y_pred)  # Kept mean_squared_error
       r2 = r2_score(y_test, y_pred)  # Kept r2_score
       # Additional metrics
       mae = mean_absolute_error(y_test, y_pred)
       rmse = mean_squared_error(y_test, y_pred, squared=False)
       explained_variance = explained_variance_score(y_test, y_pred)
       return lr, params, mse, r2, mae, rmse, explained_variance, X_train, y_train
   ```

6. Create a `job` pipeline definition
   **Jobs are the main unit of execution and monitoring in Dagster. They allow you to execute a portion of a graph of asset definitions or ops based on a schedule or an external trigger.**

   ```python

    @job
    def linear_regression_pipeline():
        data = load_data()
        split = split_data(data)
        train_model(split)
   ```

7. Open Dagster UI in the browser and navigate to the `Assets` tab -> `http://127.0.0.1:3000/assets` and click `Reload definitions`

8. Click the `default` project from the left sidebar, then click `Materialize All` to materialize all the assets

9. Click `View` pop-up dialog view to view the materialized output.

### Tracking and Model Registry with MLFlow

MLflow is an open-source platform designed to manage the machine learning (ML) lifecycle, providing tools to help with experimentation, reproducibility, and deployment of machine learning models.

1. Cancel the current Dagster run
2. Add `mlflow` to the `setup.py` file
3. Update installed package with `pip install -e ".[dev]"`
4. Run `dagster dev`
5. Open a new terminal and activate the virtual environment - `source .venv/bin/activate`
6. Run mlflow - `mlflow server --host 127.0.0.1 --port 8080`
7. Open the MLFlow UI in your browser - `http://127.0.0.1:8080` to check
8. Go back to the IDE -> `assets.py` and add on top

```python
import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
  explained_variance_score,
  mean_absolute_error,
  mean_squared_error,
  r2_score,
)


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
```

9. Define a new `asset` called `log_model`

```python
@asset
def log_model(train_model):
    """
    Log the trained model and its metrics to MLflow.

    Args:
        train_model (tuple): A tuple containing the trained model, parameters, mean squared error, R^2 score, training features, and training target.

    Returns:
        ModelInfo: Information about the logged model.
    """
    lr, params, mse, r2, mae, rmse, explained_variance, X_train, y_train = train_model
    mlflow.set_experiment("MLflow Linear Regression")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("explained_variance", explained_variance)
        mlflow.set_tag("Training Info", "Linear Regression model for Iris data")
        signature = infer_signature(X_train, lr.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_lr_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="linear-regression-quickstart",
        )
    return model_info
```

10. Update the job to

    ```python
    @job
    def linear_regression_pipeline():
        data = load_data()
        split = split_data(data)
        model = train_model(split)
        model_info = log_model(model)
    ```

11. Materialize the whole process by clicking the `default` project from the left sidebar, then click `Materialize All` to materialize all the assets

12. Open MLFlow Web by accessing `http://127.0.0.1:8080`, check the model

### Load and Serve the Model with MLFlow

1. Update the package imports

```python
import base64
from io import BytesIO
import pandas as pd

import matplotlib.pyplot as plt
from dagster import MaterializeResult, MetadataValue
```

2. Define a new `asset` for `load_and_predict`

```python
@asset
def load_and_predict(log_model, split_data):
    """
    Load the logged model and make predictions on the test data. Plot the actual vs. predicted values.

    Args:
        log_model (ModelInfo): Information about the logged model.
        split_data (tuple): A tuple containing the training and testing sets for features (X_train, X_test) and target (y_train, y_test).

    Returns:
        MaterializeResult: The result containing the plot of actual vs. predicted values as metadata.
    """
    model_info = log_model
    _, X_test, _, y_test = split_data

    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    # Ensure X_test and predictions have the same length by truncating or padding
    min_length = min(len(X_test), len(predictions))
    X_test = X_test[:min_length]
    y_test = y_test[:min_length]
    predictions = predictions[:min_length]

    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
    plt.scatter(range(len(predictions)), predictions, color="red", label="Predicted")
    plt.plot(
        range(len(predictions)), predictions, color="green", label="Regression Line"
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Target")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()

    # Convert the image to a saveable format
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    image_data = base64.b64encode(buffer.getvalue())

    # Convert the image to Markdown to preview it within Dagster
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    # Attach the Markdown content as metadata to the asset
    return MaterializeResult(metadata={"plot": MetadataValue.md(md_content)})

```

3. Update the job

```python
@job
def linear_regression_pipeline():
    data = load_data()
    split = split_data(data)
    model = train_model(split)
    model_info = log_model(model)
    load_and_predict(model_info, split)
```

4. Materialize the whole process by clicking the `default` project from the left sidebar, then click `Materialize All` to materialize all the assets

### CI/CD for ML Models with Github Actions

CI/CD stands for Continuous Integration and Continuous Deployment. CI automates code integration and testing, while CD automates code deployment to production, enabling faster and more reliable software releases.

1. Create a new repository on GitHub
2. Push the project to the repository
3. For Dev CI/CD , create a new file `.github/workflows/main.yml` with the following content

```yaml
name: CI/CD Pipeline - Dev
run-name: MLOps with Dagster and MLflow

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
env:
  DB_FLAG: CI

jobs:
  Build:
    runs-on: ubuntu-latest

    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE
      - uses: actions/checkout@v3

      # Set up Python 3.9 environment
      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      # Model building and testing
      - name: Build and Test Model
        run: echo "Building and testing the model"

      # Run the ML pipeline
      - name: Run ML pipeline
        run: echo "Running the ML pipeline"
```

4. For Staging CI/CD, create a new file `.github/workflows/staging.yml` with the following content

```yaml
name: CI/CD Pipeline - Staging
run-name: MLOps with Dagster and MLflow

on:
  push:
    branches:
      - staging
  pull_request:
    branches:
      - staging
env:
  DB_FLAG: CI

jobs:
  DeployStaging:
    runs-on: ubuntu-latest

    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE
      - uses: actions/checkout@v3

      # Set up Python 3.9 environment
      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      # Model building and testing
      - name: Build and Test Model
        run: echo "Building and testing the model"

      # Run the ML pipeline
      - name: Run ML pipeline
        run: echo "Running the ML pipeline"
```

5. For Production CI/CD, create a new file `.github/workflows/production.yml` with the following content

```yaml
name: CI/CD Pipeline - Prod
run-name: MLOps with Dagster and MLflow

on:
  push:
    branches:
      - prod
  pull_request:
    branches:
      - prod
env:
  DB_FLAG: CI

jobs:
  DeployProd:
    runs-on: ubuntu-latest

    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE
      - uses: actions/checkout@v3

      # Set up Python 3.9 environment
      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      # Model building and testing
      - name: Build and Test Model
        run: echo "Building and testing the model"

      # Run the ML pipeline
      - name: Run ML pipeline
        run: echo "Running the ML pipeline"
```

6. Push the changes to the repository
7. Do sample ML Workflow changes and push to the repository to trigger the CI/CD pipeline
