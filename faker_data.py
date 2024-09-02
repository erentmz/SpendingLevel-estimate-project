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


def create_customers(num_customers):
    customer_list = []
    for i in range(1, num_customers + 1):

        income_level = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])

        job = random.choice(jobs[income_level])

        if income_level == 'low':
            salary = fake.random_int(min=13000, max=50000)
        elif income_level == 'medium':
            salary = fake.random_int(min=51000, max=88000)
        else:
            salary = fake.random_int(min=89000, max=150000)

        age = fake.random_int(min=18, max=70)
        age_factor = 0.5 + (abs(age - 35)) / 70

        income_factor = {'low': 0.9, 'medium': 1.1, 'high': 1.3}[income_level]

        spending_score = int(100 * np.random.normal(loc=50, scale=15) * age_factor * income_factor)
        spending_score = max(0, spending_score)

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

    df = pd.DataFrame(customer_list)

    min_score = df['spending_score'].min()
    max_score = df['spending_score'].max()

    df['spending_score'] = (df['spending_score'] - min_score) / (max_score - min_score) * 99 + 1
    df['spending_score'] = df['spending_score'].round().astype(int)

    return df


data = create_customers(1000)

data.to_csv('customer_table1k.csv', index=False)
