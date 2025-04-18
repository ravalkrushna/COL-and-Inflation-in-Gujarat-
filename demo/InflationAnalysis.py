import pandas as pd

# Load dataset
df = pd.read_csv("gujarat_cost_of_living_full_dataset.csv")

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract only Dec 2023 and Dec 2024 data
dec_2023 = df[df['Date'] == '2023-12-01']
dec_2024 = df[df['Date'] == '2024-12-01']

# Ensure same districts exist in both
common_districts = sorted(set(dec_2023['District']).intersection(dec_2024['District']))
dec_2023 = dec_2023[dec_2023['District'].isin(common_districts)].set_index('District')
dec_2024 = dec_2024[dec_2024['District'].isin(common_districts)].set_index('District')

# List of expense columns to calculate inflation
expense_cols = ['Rent', 'Utilities', 'Grocery', 'Transportation', 'Healthcare', 'Education',
                'Internet', 'Entertainment', 'Fitness', 'Dining', 'Total_Monthly_Expense']

# Initialize empty dataframe
inflation_df = pd.DataFrame(index=common_districts)

# Calculate inflation percentage for each column
for col in expense_cols:
    inflation_df[col + '_Inflation (%)'] = ((dec_2024[col] - dec_2023[col]) / dec_2023[col]) * 100

# Reset index and save to CSV
inflation_df.reset_index(inplace=True)
inflation_df.rename(columns={'index': 'District'}, inplace=True)
inflation_df.to_csv("InflationAnalysis.csv", index=False)

print("✅ Inflation analysis completed and saved as 'InflationAnalysis.csv'.")

# Optional: Print top 5 districts by inflation in each category
for col in expense_cols:
    print(f"\n📈 Top 5 cities by {col} inflation:")
    print(inflation_df[['District', col + '_Inflation (%)']].sort_values(by=col + '_Inflation (%)', ascending=False).head())
