import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv("Gujarat_Cost_of_Living_and_Sustainability.csv", index_col=0)
print(df.head())
print(df.info())
print(df.describe())


sns.barplot(x="District", y="Avg Rent (₹)", data=df.sort_values("Avg Rent (₹)", ascending=False))
plt.xticks(rotation=90)
plt.title("Average Rent by District")
plt.tight_layout()
plt.show()
