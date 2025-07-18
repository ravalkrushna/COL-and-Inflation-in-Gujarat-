import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("gujarat_cost_of_living_full_dataset.csv")

# Group by District to get average monthly expenses
grouped_df = df.groupby("District").mean(numeric_only=True).reset_index()

# Columns to consider for ranking (excluding Total_Monthly_Expense for now)
columns_for_ranking = [
    'Rent', 'Utilities', 'Grocery', 'Transportation', 'Healthcare',
    'Education', 'Internet', 'Entertainment', 'Fitness', 'Dining'
]

# Normalize (scale) values using Min-Max Scaling (lower cost = better rank)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(grouped_df[columns_for_ranking])

# Create a DataFrame for scaled values
scaled_df = pd.DataFrame(scaled_data, columns=columns_for_ranking)
scaled_df['District'] = grouped_df['District']

# Compute Liveability Score (lower total = higher rank)
scaled_df['Liveability_Score'] = scaled_df[columns_for_ranking].sum(axis=1)

# Rank cities: lower score = more affordable
ranked_df = scaled_df.sort_values(by='Liveability_Score').reset_index(drop=True)

# --- Export Normalized Scores + Liveability Score ---
normalized_output = scaled_df[['District'] + columns_for_ranking + ['Liveability_Score']]
normalized_output.to_csv("City_Normalized_Scores.csv", index=False)
print("\n✅ Normalized scores and Liveability Score saved to 'City_Normalized_Scores.csv'")

# --- Export Final Rankings ---
ranked_df[['District', 'Liveability_Score']].to_csv("City_Liveability_Ranking.csv", index=False)
print("✅ Liveability Ranking saved to 'City_Liveability_Ranking.csv'")

# --- Optional Console Preview ---
print("\n🔍 Full Normalized Scores Table:")
print(normalized_output.to_string(index=False))


# --- Plot Top 10 and Bottom 10 Cities ---
def plot_ranking(df, title, top=True):
    subset = df.head(10) if top else df.tail(10)
    plt.figure(figsize=(10, 6))
    plt.barh(subset['District'], subset['Liveability_Score'], color='green' if top else 'red')
    plt.xlabel("Liveability Score (Lower is Better)")
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest rank on top
    plt.tight_layout()
    plt.show()

# Plot top 10 best cities to live
plot_ranking(ranked_df, "Top 10 Most Affordable Cities in Gujarat", top=True)

# Plot bottom 10 least affordable cities to live
plot_ranking(ranked_df, "Bottom 10 Least Affordable Cities in Gujarat", top=False)
