# Spending Level Estimation

This project uses a machine learning model to predict the spending level (low, medium, high) of customers based on their age, salary, and job.

## Overview

The project consists of the following components:

- **Data Generation:** A Python script that generates synthetic customer data with attributes like age, salary, job, and spending score.
- **Model Training:** A script that preprocesses the data, trains a Support Vector Machine (SVM) model, and saves the trained model.
- **Flask Application:** A web application that allows users to input customer information (age, salary, job) and get a prediction of their spending level.

## Getting Started

### Prerequisites

- Python 3.8 or later
- Required Python packages: `pandas`, `scikit-learn`, `imblearn`, `faker`, `Flask`, `pickle`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/erentmz/SpendingLevel-estimate-project.git

2. Install the required packages:
    ```bash
   pip install -r requirements.txt

## Running the Application

1. Navigate to the project directory:
    ```bash
   cd SpendingLevel-estimate-project

2. Run the Flask application
    ```bash
   flask run

3. Access the application in your web browser: http://127.0.0.1:5000/

## Usage
1. Enter the customer's information (age, salary, job) in the form.
2. Click the "Predict" button to get the predicted spending level.

## Contributing
Contributions are welcome! Please submit a pull request with your changes.

