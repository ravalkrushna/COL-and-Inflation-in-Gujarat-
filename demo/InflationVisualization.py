import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load processed inflation data
df = pd.read_csv("InflationAnalysis.csv", index_col=0)

# Reset index for plotting
df.reset_index(inplace=True)
df.rename(columns={'index': 'District'}, inplace=True)

# Identify all inflation columns dynamically by checking if column name contains 'Inflation'
inflation_cols = [col for col in df.columns if 'Inflation' in col]
print("Inflation columns:", inflation_cols)  # Debugging line to check column names

if inflation_cols:
    # -------------------------
    # 1. Bar Plot: Top 5 inflation-hit cities per category
    # -------------------------
    def plot_top_inflation_by_category(category_col):
        top_cities = df[['District', category_col]].sort_values(by=category_col, ascending=False).head(5)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_col, y='District', data=top_cities, palette='Reds_r')
        plt.title(f"Top 5 Cities with Highest Inflation in {category_col.replace('_Inflation','')}")
        plt.xlabel("Inflation Rate (%)")
        plt.ylabel("District")
        plt.tight_layout()
        plt.savefig(f"top5_inflation_{category_col}.png")
        plt.show()

    # Example: plot for first inflation category
    plot_top_inflation_by_category(inflation_cols[0])

    # -------------------------
    # 2. Heatmap of inflation rates across cities & categories
    # -------------------------
    heatmap_data = df.set_index('District')[inflation_cols]
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".1f")
    plt.title("City-wise Inflation Heatmap (%)")
    plt.tight_layout()
    plt.savefig("inflation_heatmap.png")
    plt.show()

    # -------------------------
    # 3. Interactive Parallel Coordinates Plot
    # -------------------------
    fig = px.parallel_coordinates(df,
        dimensions=inflation_cols,
        color=inflation_cols[0],
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={col: col.replace('_Inflation', '') for col in inflation_cols},
        title="City-wise Inflation Comparison Across Categories"
    )
    fig.write_html("inflation_parallel_coordinates.html")
    fig.show()
else:
    print("No inflation columns found in the dataset.")
