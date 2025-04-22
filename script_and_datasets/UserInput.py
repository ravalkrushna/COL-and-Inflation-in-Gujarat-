# module3_user_filter.py

def get_user_input():
    print(" Welcome to Gujarat Living Recommender ")
    salary = float(input("Enter your monthly salary (₹): "))
    print("Choose your preferred living class:")
    print("1. Lower Class\n2. Middle Class\n3. Upper Class")
    choice = input("Enter choice (1/2/3): ").strip()

    living_class = {
        '1': 'Lower Class',
        '2': 'Middle Class',
        '3': 'Upper Class'
    }.get(choice, 'Middle Class')  # fallback

    return salary, living_class

def filter_places(df, salary, living_class):
    print(f"\n📊 Filtering for: {living_class} class under ₹{salary}\n")

    # Debug: Print range of total monthly expenses
    print("💰 Expense Summary:")
    print(df['Total_Monthly_Expense'].describe())

    # Debug: Class distribution
    print("\n🏷️ Living Class Distribution:")
    print(df['Living_Class'].value_counts())

    # Apply filter
    filtered = df[
        (df['Total_Monthly_Expense'] < salary) &
        (df['Living_Class'] == living_class)
    ].copy()

    if filtered.empty:
        print("\n⚠️ No matching districts found. Try a higher salary or different class.\n")
        return filtered

    # Ranking Score
    filtered['Score'] = (
        (salary - filtered['Total_Monthly_Expense']) * 0.4 +
        filtered['Green_Cover_%'] * 0.2 +
        filtered['Public_Transport_Score_1-10'] * 0.2 +
        filtered['Walkability_Score_1-10'] * 0.2
    )

    return filtered.sort_values(by='Score', ascending=False)

# Improved categorization function
def categorize_living_class(df):
    lower_thresh = df['Total_Monthly_Expense'].quantile(0.33)
    upper_thresh = df['Total_Monthly_Expense'].quantile(0.66)

    def classify(exp):
        if exp <= lower_thresh:
            return 'Lower Class'
        elif exp <= upper_thresh:
            return 'Middle Class'
        else:
            return 'Upper Class'

    df['Living_Class'] = df['Total_Monthly_Expense'].apply(classify)
    return df

if __name__ == "__main__":
    from loadData import load_data
    from DataCleaning import (
        clean_column_names, handle_missing_values, convert_data_types,
        calculate_total_monthly_expense
    )

    df = load_data("gujarat_data.csv")
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = calculate_total_monthly_expense(df)
    df = categorize_living_class(df)

    salary, living_class = get_user_input()
    recommendations = filter_places(df, salary, living_class)

    if not recommendations.empty:
        print("\n✅ Top Recommended Districts:")
        print(recommendations[['District', 'Total_Monthly_Expense', 'Green_Cover_%',
                               'Public_Transport_Score_1-10', 'Walkability_Score_1-10']].head(10))
