import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import os

def prepare_data():
    # 1. SETUP ABSOLUTE PATHS
    # Finds the folder where train.py is located (ml-service/)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Path to your new datasets folder
    data_dir = os.path.join(base_path, 'datasets')
    
    # Path to the model folder
    model_dir = os.path.join(base_path, 'model')

    def get_data_path(filename):
        return os.path.join(data_dir, filename)

    # Automatically create the 'model' directory if it's missing
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # 2. LOAD FILES (from the datasets subfolder)
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
    id_map = {101: 'FPS_001', 102: 'FPS_002', 103: 'FPS_003', 104: 'FPS_004', 105: 'FPS_005'}
    prod_map = {1: 'Rice', 2: 'Wheat', 3: 'Sugar', 4: 'Kerosene'}
    
    dist_2023['fps_id'] = dist_2023['dealer_id'].map(id_map)
    dist_2023['commodity'] = dist_2023['product_id'].map(prod_map)
    dist_2023 = dist_2023.rename(columns={'total_demand': 'quantity_distributed'})

    # Combine 2023 and 2024 records
    df = pd.concat([
        dist_2023[['date', 'fps_id', 'commodity', 'quantity_distributed']],
        dist_2024[['date', 'fps_id', 'commodity', 'quantity_distributed']]
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

    # 5. FEATURE ENGINEERING
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df = df.ffill().bfill()
    df = pd.get_dummies(df, columns=['fps_id', 'commodity', 'city_code'])
    
    # 6. TRAINING PREPARATION
    X = df.drop(['date', 'quantity_distributed', 'major_govt_announcement'], axis=1)
    y = df['quantity_distributed']
    
    # 7. MODEL TRAINING
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    
    # 8. SAVE MODEL AND COLUMNS (to the model/ folder)
    joblib.dump(model, os.path.join(model_dir, 'ration_model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, 'feature_columns.pkl'))
    print(f"Success: Model trained and saved to {model_dir}")

if __name__ == "__main__":
    prepare_data()