import random
import faker
import pandas as pd
import numpy as np

fake = faker.Faker(locale='en_US')

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

jobs = {
    'low': low_income_jobs,
    'medium': medium_income_jobs,
    'high': high_income_jobs
}

income_ranges = {
    'low': (15000, 45000),
    'medium': (46000, 90000),
    'high': (91000, 150000)
}

job_factors = {
    'low': 1.0,
    'medium': 1.2,
    'high': 1.4
}


def calculate_age_factor(age):
    if age < 25:
        return 0.9
    elif 25 <= age < 40:
        return 1.1
    else:
        return 1.0


def calculate_spending_score(age, income, job_level):
    age_factor = calculate_age_factor(age)
    income_factor = (income - income_ranges[job_level][0]) / (income_ranges[job_level][1] - income_ranges[job_level][0])
    job_factor = job_factors[job_level]

    base_score = 30 + (40 * income_factor)

    spending_score = int(base_score * age_factor * job_factor)
    return min(100, spending_score)


def create_customers(num_customers):
    customer_list = []
    for i in range(1, num_customers + 1):
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

    return pd.DataFrame(customer_list)


data = create_customers(1000)
data.to_csv('customer_table1k.csv', index=False)
