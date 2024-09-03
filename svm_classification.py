import logging
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

logging.basicConfig(filename='svm_classification.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(data_path):
    """Loads data from a CSV file and preprocesses it for machine learning.

    Args:
        data_path (str): The path to the data file.

    Returns:
        tuple: A tuple containing the training and test data (X_train, X_test, y_train, y_test).
               or a tuple of None values if an error occurs during loading or preprocessing.

    """

    try:
        logging.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)

        # Create 'spending_level' column based on 'spending_score'
        logging.info("Creating 'spending_level' column")
        data['spending_level'] = pd.cut(data['spending_score'], bins=[0, 33, 66, 100],
                                        labels=['low', 'medium', 'high'])

        # Separate features (X) and target variable (y)
        X = data[['age', 'salary', 'job']]
        y = data['spending_level']

        # Split data into training and testing sets
        logging.info("Splitting data into training and test sets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logging.info("Data loading and preprocessing complete.")
        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        # Handle the case where the data file is not found
        logging.error(f"Data file not found. {data_path}")
        return None, None, None, None

    except Exception as e:
        # Handle other exceptions that may occur during loading or preprocessing
        logging.error(f"Error during data loading/preprocessing: {e}")
        return None, None, None, None


def create_pipeline():
    """Creates pipeline for data preprocessing and model training.

    Returns:
        Pipeline: A scikit-learn Pipeline object or None if an error occurs.
    """
    try:
        logging.info("Creating pipeline...")

        # Define numerical features for scaling
        numeric_features = ['age', 'salary']
        # Create a pipeline for numerical feature preprocessing
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Define categorical features for one-hot encoding
        categorical_features = ['job']
        # Create a pipeline for categorical feature preprocessing
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine numerical and categorical transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the main pipeline with preprocessing and the SVM classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf'))
        ])

        logging.info("Pipeline created successfully.")
        return pipeline

    except Exception as e:
        # Handle exceptions during pipeline creation
        logging.error(f"Error creating pipeline: {e}")
        return None


def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """
    Trains the model using the provided pipeline and evaluates its performance.

    Args:
        pipeline (Pipeline): The scikit-learn pipeline to use for training.
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Testing data labels.

    Returns:
        Pipeline: The trained scikit-learn Pipeline object,
                  or None if an error occurs.
    """
    try:
        # Define hyperparameters to search
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': [0.001, 0.01, 0.1, 1],
            'classifier__class_weight': [None, 'balanced']
        }

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        logging.info("Making predictions and evaluating...")
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

        logging.info("Model training and evaluation complete.")
        return best_model  # Return the best model found by GridSearchCV

    except Exception as e:
        # Handle exceptions during training or evaluation
        logging.error(f"Error during model training/evaluation: {e}")
        return None


def save_model(model, filename):
    """Saves the trained model to a file.

       Args:
           model: The trained machine learning model.
           filename (str): The filename to save the model to.
    """
    try:
        # Open the file in binary write mode
        logging.info(f"Saving model to {filename}")
        with open(filename, 'wb') as file:
            # Use pickle to serialize and the save the model to the file
            pickle.dump(model, file)

    except Exception as e:
        # Handle exceptions that may occur during saving
        logging.error(f"Error during model saving: {e}")


def load_model(filename):
    """Loads a trained model from a file.

        Args:
            filename (str): The filename of the saved model.

        Returns:
            The loaded machine learning model, or None if an error occurs.
    """
    try:
        # Open the file in binary read mode
        logging.info(f"Loading model from {filename}")
        with open(filename, 'rb') as file:
            # Use pickle to load and deserialize the model from file
            return pickle.load(file)

    except FileNotFoundError:
        # Handle the case where the model file is not found
        logging.error(f"Model file not found. {filename}")
        return None

    except Exception as e:
        # Handle other exceptions that may occur during loading
        logging.error(f"Error during model loading: {e}")
        return None


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/customer_table1k.csv')
    # Proceed only if data loading was successful
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        model_pipeline = create_pipeline()
        # Proceed only if pipeline creation was successful
        if model_pipeline is not None:
            trained_model = train_and_evaluate_model(model_pipeline, X_train, X_test, y_train, y_test)
            # Proceed to save the model only if training was successful
            if trained_model is not None:
                save_model(trained_model, 'models/trained_svm_model.pkl')