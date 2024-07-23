import base64
from io import BytesIO

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from dagster import MaterializeResult, MetadataValue, asset, job

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


@asset
def load_data():
    """
    Load the Iris dataset.

    Returns:
      tuple: A tuple containing the features (X) and the target (y) of the Iris dataset.
    """
    X, y = datasets.load_iris(return_X_y=True)  # Changed to load_iris
    return X, y


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


@job
def linear_regression_pipeline():
    data = load_data()
    split = split_data(data)
    model = train_model(split)
    model_info = log_model(model)
    load_and_predict(model_info, split)
