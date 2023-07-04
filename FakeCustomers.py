import pandas as pd
from faker import Faker

fake = Faker()

data = {
    'FirstName': [fake.first_name() for _ in range(1000)],
    'LastName': [fake.last_name() for _ in range(1000)],
    'Email': [fake.email() for _ in range(1000)],
    'Phone': [fake.phone_number() for _ in range(1000)],
    'Address': [fake.address().replace('\n', ', ') for _ in range(1000)],  # replace newline characters for easier handling
    'City': [fake.city() for _ in range(1000)],
    'State': [fake.state() for _ in range(1000)],
    'ZipCode': [fake.zipcode() for _ in range(1000)]
}

df = pd.DataFrame(data)

# Save the data to an Excel file
df.to_excel('fake_data.xlsx', index=False)
