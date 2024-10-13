import csv
import pandas as pd

with open("employee.csv", mode='w', newline='') as f:
    wr = csv.writer(f)

    # Writing the header row
    heading = ["Emp_ID", "Name", "Salary", "Project working on", "Dept_ID"]
    wr.writerow(heading)

    # Input employee data and write to the CSV
    for i in range(5):  
        emp_id = int(input("Enter employee ID: "))
        e_name = input("Enter employee name: ")
        salary = int(input("Enter employee salary: "))
        project = input("Enter project on which employee is currently working: ")
        dept_id = int(input("Enter department ID: "))
        
        # Writing employee data into CSV file
        val = [emp_id, e_name, salary, project, dept_id]
        wr.writerow(val)
        print(f"Details of employee {i+1} stored successfully!")

# Now read the data from CSV and load into a DataFrame
df = pd.read_csv("employee.csv")

# Display the data to show the association between independent and dependent variables
print("\nData from CSV:\n", df)

# Let's focus on the relationship between 'Salary' (dependent) and the other variables (independent)
# For demonstration, we'll show how the salary varies with other attributes
X = df[['Emp_ID', 'Name', 'Project working on', 'Dept_ID']]  # Independent variables
y = df['Salary']  # Dependent variable

# Display independent and dependent variables
print("\nIndependent Variables (Features):\n", X)
print("\nDependent Variable (Target - Salary):\n", y)

# Show a quick analysis, for example, head of the independent variables and salaries
print("\nFirst few records of independent variables:\n", X.head())
print("\nFirst few salaries:\n", y.head())
f.close()
