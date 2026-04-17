import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import os

def prepare_data():
    # 1. SETUP ABSOLUTE PATHS
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_path, 'datasets')
    model_dir = os.path.join(base_path, 'model')

    def get_data_path(filename):
        return os.path.join(data_dir, filename)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # 2. LOAD FILES
    try:
        dist_2024 = pd.read_csv(get_data_path('historical_ration_distribution.csv'))
        dist_2023 = pd.read_csv(get_data_path('forecasting_dataset_2023.csv'))
        weather = pd.read_csv(get_data_path('weather_climate_data.csv'))
        socio = pd.read_csv(get_data_path('socio_economic_data.csv'))
        temporal = pd.read_csv(get_data_path('temporal_event_data.csv'))
        fps_info = pd.read_csv(get_data_path('fps_information.csv'))
        print(f"Success: All CSV files loaded from {data_dir}")
    except FileNotFoundError as e:
        print(f"Error: Could not find {e.filename}. Check if it is inside: {data_dir}")
        return

    # 3. HARMONIZE 2023 AND 2024 DATA
    # Use the cleaned version we just created
    dist_2023 = pd.read_csv(get_data_path('forecasting_dataset_2023_cleaned.csv'))
    
    # Combine 2023 and 2024 records - KEEPING beneficiary_class
    df = pd.concat([
        dist_2023[['date', 'fps_id', 'commodity', 'beneficiary_class', 'quantity_distributed']],
        dist_2024[['date', 'fps_id', 'commodity', 'beneficiary_class', 'quantity_distributed']]
    ])
    
    # 4. MERGE EXTERNAL FACTORS
    df['date'] = pd.to_datetime(df['date'])
    weather['date'] = pd.to_datetime(weather['date'])
    socio['date'] = pd.to_datetime(socio['date'])
    temporal['date'] = pd.to_datetime(temporal['date'])

    df = df.merge(fps_info[['fps_id', 'city_code']], on='fps_id', how='left')
    df = df.merge(weather, on=['date', 'city_code'], how='left')
    df = df.merge(socio, on='date', how='left')
    df = df.merge(temporal, on='date', how='left')

    # --- INJECTING MEMORY (LAG FEATURES) ---
    # This helps the model see what happened last month to predict this month
    df = df.sort_values(['fps_id', 'commodity', 'date'])
    df['prev_month_demand'] = df.groupby(['fps_id', 'commodity'])['quantity_distributed'].shift(1)
    df = df.fillna(0) 

    # 5. FEATURE ENGINEERING
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df = df.ffill().bfill()
    # ADD beneficiary_class here!
    df = pd.get_dummies(df, columns=['fps_id', 'commodity', 'city_code', 'beneficiary_class'])
    
    # 6. TRAINING PREPARATION
    X = df.drop(['date', 'quantity_distributed', 'major_govt_announcement'], axis=1)
    y = df['quantity_distributed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. MODEL TRAINING (With Optimized Hyperparameters)
    model = XGBRegressor(
        n_estimators=500,     
        max_depth=8,          
        learning_rate=0.05,   
        subsample=0.8,        
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # 8. ACCURACY CHECK
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- MODEL ACCURACY REPORT ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} KG")
    print(f"Accuracy (R2 Score): {r2:.4f}")
    print(f"Average Error Percentage (MAPE): {mape * 100:.2f}%")
    print(f"Model Accuracy: {100 - (mape * 100):.2f}%")
    print("------------------------------\n")
    
    # 9. SAVE MODEL AND COLUMNS
    joblib.dump(model, os.path.join(model_dir, 'ration_model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, 'feature_columns.pkl'))
    print(f"Success: Model trained and saved to {model_dir}")

if __name__ == "__main__":
    prepare_data()