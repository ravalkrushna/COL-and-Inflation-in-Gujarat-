import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Gujarat_Cost_of_Living_and_Sustainability.csv')

# Data Cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing numeric values with district averages
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Add calculated columns
df['Total Monthly Cost'] = df['Avg Rent (1BHK) (₹)'] + df['Utilities (₹)'] + df['Groceries (₹)']
df['Affordability Score'] = df['Monthly Income (Avg) (₹)'] / df['Total Monthly Cost']

# Save cleaned data
df.to_csv('cleaned_gujarat_data.csv', index=False)

# Basic Visualization
plt.figure(figsize=(12, 6))
df_sorted = df.sort_values('Total Monthly Cost', ascending=False)
plt.barh(df_sorted['District'], df_sorted['Total Monthly Cost'], color='skyblue')
plt.xlabel('Total Monthly Cost (₹)')
plt.title('Cost of Living by District in Gujarat')
plt.tight_layout()
plt.savefig('cost_comparison.png')
plt.show()