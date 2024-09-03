import random
import faker
import pandas as pd
import numpy as np
import logging

logging.basicConfig(filename='create_data.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

fake = faker.Faker(locale='en_US')

# Define lists of jobs for different income levels
low_income_jobs = ['Cashier', 'Waiter', 'Fast Food Worker',
                   'Retail Salesperson', 'Janitor',
                   'Home Health Aide', 'Security Guard',
                   'Customer Service Representative',
                   'Taxi Driver', 'Farm Worker', 'Teaching Assistant',
                   'Dishwashers', 'Housekeeper']

medium_income_jobs = ['Teacher', 'Police Officer', 'Firefighter',
                      'Nurse', 'Electrician', 'Plumber',
                      'Carpenter', 'Accountant', 'Social Worker',
                      'Graphic Designer', 'Human Resources']

high_income_jobs = ['Software Engineer', 'Doctor', 'Lawyer',
                    'Financial Analyst', 'Marketing Manager',
                    'Sales Manager', 'Business Analyst', 'Project Manager',
                    'Data Scientist', 'CEO', 'Physician', 'Dentist',
                    'Data Engineer', 'Engineer', 'Architect', 'Gynecologist']

# Create a dictionary to map income levels to job lists
jobs = {
    'low': low_income_jobs,
    'medium': medium_income_jobs,
    'high': high_income_jobs
}

# Define income ranges for each income level
income_ranges = {
    'low': (15000, 45000),
    'medium': (46000, 90000),
    'high': (91000, 150000)
}

# Define job factors that influence spending score
job_factors = {
    'low': 1.0,
    'medium': 1.2,
    'high': 1.4
}


def calculate_age_factor(age):
    """Calculates an age factor for spending score calculation.

    Args:
        age (int): The age of the customer.

    Returns:
        float: The age factor based on the provided age.
            Returns 1.0 if an invalid age is provided.
    """
    try:
        logging.debug(f"Calculating age factor for age: {age}")
        if age < 25:
            return 0.9
        elif 25 <= age < 40:
            return 1.1
        else:
            return 1.0
    except TypeError as e:
        logging.error(f"Error: TypeError occurred in age calculation: {e}")
        return 1.0  # Return a default value


def calculate_spending_score(age, income, job_level):
    """Calculates a spending score based on age, income, and job level.

    Args:
        age (int): The age of the customer.
        income (int): The income of the customer.
        job_level (str): The income level of the customer's job ('low', 'medium', or 'high').

    Returns:
        int: The calculated spending score (between 0 and 100).
            Returns 0 if an error occurs during calculation.
    """
    try:
        logging.debug(f"Calculating spending score for age: {age}, income: {income}, job_level: {job_level}")
        age_factor = calculate_age_factor(age)
        income_factor = (income - income_ranges[job_level][0]) / (
                income_ranges[job_level][1] - income_ranges[job_level][0])
        job_factor = job_factors[job_level]

        base_score = 30 + (40 * income_factor)

        spending_score = int(base_score * age_factor * job_factor)
        return min(100, spending_score)
    except KeyError as e:
        logging.error(f"Error: KeyError occurred during income calculation: {e}")
        return 0
    except TypeError as e:
        logging.error(f"Error: TypeError occurred during income calculation: {e} ")
        return 0


def create_customers(num_customers):
    """Creates a DataFrame containing customer data.

    Args:
        num_customers (int): The number of customers to create.

    Returns:
        pandas.DataFrame: A DataFrame containing customer data, or an empty DataFrame
                         if an error occurs during creation.
    """
    try:
        logging.info(f"Creating {num_customers} customer records...")
        customer_list = []
        for i in range(1, num_customers + 1):
            try:
                income_level = np.random.choice(['low', 'medium', 'high'], p=[0.25, 0.5, 0.25])
                job = random.choice(jobs[income_level])
                salary = fake.random_int(*income_ranges[income_level])
                age = fake.random_int(min=18, max=70)
                spending_score = calculate_spending_score(age, salary, income_level)

                customer = {
                    'customerId': i,
                    'name_surname': fake.name(),
                    'age': age,
                    'gender': fake.random_element(elements=('Male', 'Female')),
                    'job': job,
                    'salary': salary,
                    'spending_score': spending_score
                }
                customer_list.append(customer)
                logging.debug(f"Created customer record: {customer}")
            except Exception as e:
                logging.error(f"Error creating customer: {e}")
        logging.info(f"{len(customer_list)} customer records created successfully.")
        return pd.DataFrame(customer_list)
    except Exception as e:
        logging.error(f"Error creating customer DataFrame: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error


data = create_customers(1000)
data.to_csv('customer_table1k.csv', index=False)
