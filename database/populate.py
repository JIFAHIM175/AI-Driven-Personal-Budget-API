import pandas as pd
import psycopg2

# Load the dataset from the CSV file.
# Adjust the file path if necessary.
df = pd.read_csv('database\\11 march 2025.csv')

# Standardize column names (convert to lowercase and remove extra spaces)
df.columns = [col.strip().lower() for col in df.columns]

# Optional: display first few rows to check the data
print(df.head())

# Connect to your PostgreSQL database.
conn = psycopg2.connect(
    host="localhost",
    database="budget_data",
    user="postgres",
    password="postgres"
)
cursor = conn.cursor()

# Insert each row into the real_budget table.
for _, row in df.iterrows():
    cursor.execute('''
        INSERT INTO real_budget (date, category, amount)
        VALUES (%s, %s, %s)
    ''', (row['date'], row['category'], row['amount']))

conn.commit()
cursor.close()
conn.close()

print("Data inserted successfully into the real_budget table.")








