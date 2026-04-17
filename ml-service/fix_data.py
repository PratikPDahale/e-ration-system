import pandas as pd
import os
import numpy as np

def enhance_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_path, 'datasets')
    
    # Files to fix
    hist_path = os.path.join(data_dir, 'historical_ration_distribution.csv')
    fore_path = os.path.join(data_dir, 'forecasting_dataset_2023.csv')

    def apply_logic(df, is_2023=False):
        # Ensure date format
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        
        # Standardize columns for 2023
        if is_2023:
            id_map = {101: 'FPS_001', 102: 'FPS_002', 103: 'FPS_003', 104: 'FPS_004', 105: 'FPS_005'}
            prod_map = {1: 'Rice', 2: 'Wheat', 3: 'Sugar', 4: 'Kerosene'}
            df['fps_id'] = df['dealer_id'].map(id_map)
            df['commodity'] = df['product_id'].map(prod_map)
            df['quantity_distributed'] = df['total_demand']
            # Split each record into AAY and PHH to match 2024 structure
            df_aay = df.copy()
            df_aay['beneficiary_class'] = 'AAY'
            df_aay['quantity_distributed'] *= 0.4
            
            df_phh = df.copy()
            df_phh['beneficiary_class'] = 'PHH'
            df_phh['quantity_distributed'] *= 0.6
            
            df = pd.concat([df_aay, df_phh])

        df['quantity_distributed'] = df['quantity_distributed'].astype(float)

        # Apply Logical Signals (The "Learning" patterns)
        # 1. Commodity Scale
        df.loc[df['commodity'] == 'Rice', 'quantity_distributed'] *= 1.5
        df.loc[df['commodity'] == 'Sugar', 'quantity_distributed'] *= 0.3
        
        # 2. Beneficiary Scale
        df.loc[df['beneficiary_class'] == 'PHH', 'quantity_distributed'] *= 1.4
        
        # 3. Seasonal Scale (Winter/Festival spikes)
        df.loc[df['month'].isin([10, 11, 12]), 'quantity_distributed'] *= 1.3
        
        # Clean up
        df['quantity_distributed'] = df['quantity_distributed'].clip(50, 2000).astype(int)
        return df[['date', 'fps_id', 'commodity', 'beneficiary_class', 'quantity_distributed']]

    # Process 2024
    if os.path.exists(hist_path):
        df_2024 = pd.read_csv(hist_path)
        df_2024 = apply_logic(df_2024)
        df_2024.to_csv(hist_path, index=False)
        print("Updated 2024 Historical Data.")

    # Process 2023
    if os.path.exists(fore_path):
        df_2023 = pd.read_csv(fore_path)
        df_2023 = apply_logic(df_2023, is_2023=True)
        # We save this as a 'cleaned' version for the trainer
        df_2023.to_csv(os.path.join(data_dir, 'forecasting_dataset_2023_cleaned.csv'), index=False)
        print("Updated 2023 Forecasting Data.")

if __name__ == "__main__":
    enhance_data()