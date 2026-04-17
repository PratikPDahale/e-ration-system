import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def generate_visuals():
    # Load Model and Columns
    model_path = 'model/ration_model.pkl'
    cols_path = 'model/feature_columns.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found! Run train.py first.")
        return

    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)

    # 1. Plot Feature Importance
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title('Top 10 Factors Influencing Ration Demand')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance_actual.png')
    print("Saved: feature_importance_actual.png")

    # 2. Demand Trends Visualization
    # (Loading historical data for trend analysis)
    df = pd.read_csv('datasets/historical_ration_distribution.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='month', y='quantity_distributed', hue='commodity', marker='o')
    plt.title('Ration Distribution Trends (Model Training Data)')
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    plt.savefig('demand_trends_actual.png')
    print("Saved: demand_trends_actual.png")

if __name__ == "__main__":
    generate_visuals()