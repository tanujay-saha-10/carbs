import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('observations.db')

# Execute the query and fetch results
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", conn)

# Display the table names
print("The tables in this database are:")
print(tables['name'].tolist())


df = pd.read_sql_query("SELECT * FROM observations", conn)
print("\nThe columns in the observations table are:")
print(df.columns)

k = 0
print("Row " + str(k) + " of observation table:")
for item in df.columns:
    print('\n')
    print(item)
    print(df[item][k])


# Close the connection
conn.close()
