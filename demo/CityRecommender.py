# CityRecommender.py
import pandas as pd

def get_user_input():
    print("🔎 Welcome to Gujarat City Recommendation System")
    salary = float(input("💰 Enter your monthly salary (₹): "))
    print("Choose your preferred living class:\n1. Lower Class\n2. Middle Class\n3. Upper Class")
    living_class = {
        '1': 'Lower Class',
        '2': 'Middle Class',
        '3': 'Upper Class'
    }.get(input("Enter choice (1/2/3): ").strip(), 'Middle Class')
    return salary, living_class

def categorize_living_class(df):
    lower = df['Total_Monthly_Expense'].quantile(0.33)
    upper = df['Total_Monthly_Expense'].quantile(0.66)
    def classify(exp):
        if exp <= lower: return 'Lower Class'
        elif exp <= upper: return 'Middle Class'
        else: return 'Upper Class'
    df['Living_Class'] = df['Total_Monthly_Expense'].apply(classify)
    return df

def score_and_recommend(df, salary, living_class):
    print(f"\n🎯 Filtering cities for: {living_class} class within ₹{salary} budget")
    filtered = df[(df['Total_Monthly_Expense'] <= salary) & (df['Living_Class'] == living_class)].copy()

    if filtered.empty:
        print("⚠️ No cities match your filters. Try increasing salary or changing class.")
        return filtered

    # Scoring based on how affordable + low inflation + rent + healthcare
    filtered['Score'] = (
        (salary - filtered['Total_Monthly_Expense']) * 0.5 -
        filtered['Rent_Inflation (%)'] * 5 -
        filtered['Healthcare_Inflation (%)'] * 5
    )

    return filtered.sort_values(by='Score', ascending=False)

if __name__ == "__main__":
    # Load both datasets
    inflation_df = pd.read_csv("InflationAnalysis.csv")
    cost_df = pd.read_csv("gujarat_cost_of_living_full_dataset.csv")

    # ✅ Get latest date entry per district
    cost_df['Date'] = pd.to_datetime(cost_df['Date'])
    cost_df = cost_df.sort_values('Date').groupby('District').tail(1)

    # Merge on District
    df = pd.merge(cost_df, inflation_df, on='District', how='inner')

    # Classify and recommend
    df = categorize_living_class(df)
    salary, living_class = get_user_input()
    recommendations = score_and_recommend(df, salary, living_class)

    if not recommendations.empty:
        print("\n🏆 Top Recommended Cities:")
        print(recommendations[['District', 'Total_Monthly_Expense', 'Rent_Inflation (%)',
                               'Healthcare_Inflation (%)', 'Score']].head(5))
