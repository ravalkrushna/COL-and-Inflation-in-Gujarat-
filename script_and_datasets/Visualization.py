# module4_visualization.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_total_expense_by_district(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Total_Monthly_Expense', y='District', data=df.sort_values('Total_Monthly_Expense', ascending=False))
    plt.title('District-wise Total Monthly Expense')
    plt.xlabel('Total Monthly Expense (₹)')
    plt.ylabel('District')
    plt.tight_layout()
    plt.show()

def plot_green_cover(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Green_Cover_%', y='District', data=df.sort_values('Green_Cover_%', ascending=False))
    plt.title('District-wise Green Cover (%)')
    plt.xlabel('Green Cover %')
    plt.ylabel('District')
    plt.tight_layout()
    plt.show()

def plot_transport_and_walkability(df):
    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=df, x='Public_Transport_Score_1-10', y='Walkability_Score_1-10', hue='District', s=100)
    plt.title('Transport vs Walkability by District')
    plt.xlabel('Public Transport Score (1-10)')
    plt.ylabel('Walkability Score (1-10)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_living_class_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Living_Class', palette='pastel')
    plt.title('Distribution of Districts by Living Class')
    plt.xlabel('Living Class')
    plt.ylabel('Number of Districts')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from loadData import load_data
    from DataCleaning import (
        clean_column_names, handle_missing_values, convert_data_types,
        calculate_total_monthly_expense
    )
    from UserInput import categorize_living_class

    df = load_data("gujarat_data.csv")
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = calculate_total_monthly_expense(df)
    df = categorize_living_class(df)

    plot_total_expense_by_district(df)
    plot_green_cover(df)
    plot_transport_and_walkability(df)
    plot_living_class_distribution(df)
