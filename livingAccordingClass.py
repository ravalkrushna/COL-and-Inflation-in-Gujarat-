import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Gujarat_Cost_of_Living_and_Sustainability.csv", index_col=0)

# Define classification function
def classify_city(row):
    if row["Monthly Income (₹)"] >= 22000:
        return "Upper Class"
    elif row["Monthly Income (₹)"] >= 19000:
        return "Middle Class"
    else:
        return "Lower Class"

# Apply the classification to the dataframe
df["Economic Class"] = df.apply(classify_city, axis=1)

# Calculate Cost Index
df["Cost Index"] = df[["Avg Rent (₹)", "Utilities (₹)", "Groceries (₹)", "Transport (₹)", "Healthcare (₹)"]].sum(axis=1)

# Calculate Savings (₹)
df["Savings (₹)"] = df["Monthly Income (₹)"] - df["Cost Index"]

# Calculate Affordability (%)
df["Affordability (%)"] = (df["Savings (₹)"] / df["Monthly Income (₹)"]) * 100

# Scatter Plot: Monthly Income vs Cost Index
plt.figure(figsize=(10,6))
sns.scatterplot(x="Monthly Income (₹)", y="Cost Index", data=df, hue="Economic Class", palette="coolwarm", s=100)
plt.title("Monthly Income vs Cost Index (Classified by Economic Class)")
plt.xlabel("Monthly Income (₹)")
plt.ylabel("Cost Index (₹)")
plt.legend(title="Economic Class")
plt.tight_layout()
plt.show()

# Scatter Plot: Avg Rent vs Groceries
plt.figure(figsize=(10,6))
sns.scatterplot(x="Avg Rent (₹)", y="Groceries (₹)", data=df, hue="Economic Class", palette="coolwarm", s=100)
plt.title("Average Rent vs Groceries (Classified by Economic Class)")
plt.xlabel("Avg Rent (₹)")
plt.ylabel("Groceries (₹)")
plt.legend(title="Economic Class")
plt.tight_layout()
plt.show()

# Scatter Plot: Green Cover vs Affordability
plt.figure(figsize=(10,6))
sns.scatterplot(x="Green Cover (%)", y="Affordability (%)", data=df, hue="Economic Class", palette="coolwarm", s=100)
plt.title("Green Cover vs Affordability (%)")
plt.xlabel("Green Cover (%)")
plt.ylabel("Affordability (%)")
plt.legend(title="Economic Class")
plt.tight_layout()
plt.show()
