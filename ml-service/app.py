from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Setup Absolute Paths
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'model', 'ration_model.pkl')
columns_path = os.path.join(base_path, 'model', 'feature_columns.pkl')

# Load the trained model and feature list
model = joblib.load(model_path)
model_features = joblib.load(columns_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Received from Spring Boot
        
        # Create a DataFrame with all 0s using the original feature columns
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        
        # Fill in the basic numeric values
        input_df['month'] = int(data.get('month', 1))
        input_df['year'] = int(data.get('year', 2024))
        
        # Handle One-Hot Encoding for FPS ID and Commodity
        fps_col = f"fps_id_{data.get('fps_id')}"
        comm_col = f"commodity_{data.get('commodity')}"
        
        if fps_col in input_df.columns:
            input_df[fps_col] = 1
        if comm_col in input_df.columns:
            input_df[comm_col] = 1

        # Run Prediction
        prediction = model.predict(input_df)
        
        return jsonify({
            'status': 'success',
            'predicted_quantity': float(prediction[0])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Running on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)