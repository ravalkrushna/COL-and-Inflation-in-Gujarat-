# module2_clean_data.py

import pandas as pd

def clean_column_names(df):
    df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("₹", "INR").replace("/", "_per_") for col in df.columns]
    return df

def handle_missing_values(df):
    # Drop rows where essential data is missing (or fill if few)
    df = df.dropna()
    return df

def convert_data_types(df):
    numeric_cols = [col for col in df.columns if col not in ['District']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_total_monthly_expense(df):
    df['Total_Monthly_Expense'] = (
        df['Avg_Rent_1BHK_INR'] + df['Utilities_INR'] + df['Groceries_INR'] +
        df['Dining_Out_INR'] + df['Transport_Monthly_Pass_INR'] +
        df['Internet_100Mbps_INR'] + df['Fuel_INR_per_liter'] * 20  # approx monthly consumption
    )
    return df

def categorize_living_class(df):
    def classify(row):
        income = row['Monthly_Income_Avg_INR']
        if income <= 18000:
            return 'Lower Class'
        elif income <= 30000:
            return 'Middle Class'
        else:
            return 'Upper Class'
    
    df['Living_Class'] = df.apply(classify, axis=1)
    return df

if __name__ == "__main__":
    from loadData import load_data

    df = load_data("gujarat_data.csv")
    if df is not None:
        df = clean_column_names(df)
        df = handle_missing_values(df)
        df = convert_data_types(df)
        df = calculate_total_monthly_expense(df)
        df = categorize_living_class(df)

        print("\n📦 Processed Data Sample:")
        print(df[['District', 'Total_Monthly_Expense', 'Monthly_Income_Avg_INR', 'Living_Class']].head())
