import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data['spending_level'] = pd.cut(data['spending_score'], bins=[0, 33, 66, 100],
                                    labels=['low', 'medium', 'high'])
    X = data[['age', 'salary', 'job']]
    y = data['spending_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def create_pipeline():
    numeric_features = ['age', 'salary']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['job']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', C=1))
    ])
    return pipeline


def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Doğruluk: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    return pipeline


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('customer_table1k.csv')
    model_pipeline = create_pipeline()
    trained_model = train_and_evaluate_model(model_pipeline, X_train, X_test, y_train, y_test)
    save_model(trained_model, 'models/trained_svm_model.pkl')
