import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==============================
# 1. LOAD DATA
# ==============================

print("Loading raw data...")
df = pd.read_csv("raw_finance_data.csv")

# ==============================
# 2. DATA CLEANING
# ==============================

print("Cleaning data...")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert Amount to numeric
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Remove rows with invalid Date or Amount
df = df.dropna(subset=['Date', 'Amount'])

# Remove duplicates
df = df.drop_duplicates()

# Fill missing categories or payment mode
df['Category'] = df['Category'].fillna("Unknown")
df['Payment_Mode'] = df['Payment_Mode'].fillna("Unknown")

# Standardize Type column
df['Type'] = df['Type'].str.capitalize()

# ==============================
# 3. FEATURE ENGINEERING
# ==============================

print("Creating new features...")

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')
df['Weekday'] = df['Date'].dt.day_name()

# Separate Income and Expense columns
df['Income'] = np.where(df['Type'] == 'Income', df['Amount'], 0)
df['Expense'] = np.where(df['Type'] == 'Expense', df['Amount'], 0)

# ==============================
# 4. MONTHLY SUMMARY
# ==============================

print("Generating monthly summary...")

monthly_summary = df.groupby(['Year', 'Month', 'Month_Name']).agg({
    'Income': 'sum',
    'Expense': 'sum'
}).reset_index()

monthly_summary['Savings'] = monthly_summary['Income'] - monthly_summary['Expense']

monthly_summary['Savings_Percentage'] = np.where(
    monthly_summary['Income'] > 0,
    (monthly_summary['Savings'] / monthly_summary['Income']) * 100,
    0
)

# Sort properly by Year and Month
monthly_summary = monthly_summary.sort_values(['Year', 'Month'])

# ==============================
# 5. CATEGORY SUMMARY
# ==============================

print("Generating category summary...")

category_summary = df.groupby('Category')['Expense'].sum().reset_index()
category_summary = category_summary.sort_values(by='Expense', ascending=False)

# ==============================
# 6. FORECAST NEXT MONTH EXPENSE
# ==============================

print("Predicting next month expense...")

if len(monthly_summary) > 1:
    monthly_summary['Month_Index'] = np.arange(len(monthly_summary))

    X = monthly_summary[['Month_Index']]
    y = monthly_summary['Expense']

    model = LinearRegression()
    model.fit(X, y)

    next_month_index = [[len(monthly_summary)]]
    predicted_expense = model.predict(next_month_index)[0]

else:
    predicted_expense = 0

# Create forecast dataframe
forecast_data = pd.DataFrame({
    "Predicted_Next_Month_Expense": [round(predicted_expense, 2)]
})

# ==============================
# 7. EXPORT FILES
# ==============================

print("Exporting processed files...")

df.to_csv("processed_transactions.csv", index=False)
monthly_summary.to_csv("monthly_summary.csv", index=False)
category_summary.to_csv("category_summary.csv", index=False)
forecast_data.to_csv("forecast_data.csv", index=False)

print("====================================")
print("Processing Complete ✅")
print("Predicted Next Month Expense:", round(predicted_expense, 2))
print("Files Created:")
print("- processed_transactions.csv")
print("- monthly_summary.csv")
print("- category_summary.csv")
print("- forecast_data.csv")
print("====================================")