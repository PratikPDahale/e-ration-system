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
    model_dir = os.path.join(base_path, 'model')
    data_dir = os.path.join(base_path, 'datasets')
    
    model_path = os.path.join(model_dir, 'ration_model.pkl')
    cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    data_path = os.path.join(data_dir, 'historical_ration_distribution.csv')
    
    # 2. LOAD DATA & MODEL
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train.py first!")
        return
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    df = pd.read_csv(data_path)
    
    # Pre-process for plotting
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    print("Generating updated pictorial representations...")

    # --- CHART 1: FEATURE IMPORTANCE (The "Why") ---
    # We filter for the top 10 features to keep the report clean
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    
    # Create a DataFrame for easier sorting
    feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(12)

    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('Key Drivers of Monthly Ration Demand', fontsize=16, pad=15)
    plt.xlabel('Importance Score (XGBoost Weight)')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.savefig('report_feature_importance.png')
    plt.close()

    # --- CHART 2: MONTHLY DEMAND TRENDS (The "Logic") ---
    # CRITICAL: We aggregate by month here to match the 1000+ KG scale
    monthly_df = df.groupby(['year', 'month', 'commodity'])['quantity_distributed'].sum().reset_index()
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=monthly_df, x='month', y='quantity_distributed', 
                 hue='commodity', marker='o', linewidth=2.5, markersize=8)
    
    plt.title('Historical Monthly Demand Patterns (Scale: 1000+ KG)', fontsize=16, pad=15)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel('Total Monthly Quantity (KG)')
    plt.xlabel('Month of Year')
    plt.legend(title='Commodity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('report_demand_trends.png')
    plt.close()

    print("\n--- SUCCESS ---")
    print(f"Visuals updated for monthly scale (~{monthly_df['quantity_distributed'].mean():.2f} KG average)")
    print("Files saved:")
    print("1. report_feature_importance.png")
    print("2. report_demand_trends.png")

if __name__ == "__main__":
    generate_visuals()