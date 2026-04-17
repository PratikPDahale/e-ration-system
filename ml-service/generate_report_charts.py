import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set visual style for the report
sns.set_theme(style="whitegrid")

def generate_visuals():
    # 1. SETUP PATHS
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'model', 'ration_model.pkl')
    cols_path = os.path.join(base_path, 'model', 'feature_columns.pkl')
    data_path = os.path.join(base_path, 'datasets', 'historical_ration_distribution.csv')
    
    # 2. LOAD DATA & MODEL
    if not os.path.exists(model_path):
        print("Error: Model file not found. Run train.py first!")
        return
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    print("Generating pictorial representations...")

    # --- CHART 1: FEATURE IMPORTANCE (The "Why") ---
    # Shows which factors (Commodity, Month, etc.) drive the accuracy
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title('Top 10 Factors Influencing Ration Demand', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('report_feature_importance.png')
    

    # --- CHART 2: DEMAND TRENDS (The "Logic") ---
    # Shows the seasonal patterns the model learned
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='month', y='quantity_distributed', hue='commodity', marker='o')
    plt.title('Seasonal Demand Trends: Rice vs Wheat vs Sugar', fontsize=14)
    plt.xticks(range(1, 13))
    plt.ylabel('Quantity (KG)')
    plt.grid(True, alpha=0.3)
    plt.savefig('report_demand_trends.png')
    

    print("--- SUCCESS ---")
    print("Files saved in ml-service folder:")
    print("1. report_feature_importance.png")
    print("2. report_demand_trends.png")

if __name__ == "__main__":
    generate_visuals()