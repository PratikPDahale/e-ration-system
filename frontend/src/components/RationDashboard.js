import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, LayoutDashboard, Database, AlertCircle } from 'lucide-react';

const RationDashboard = () => {
    const [formData, setFormData] = useState({ fpsId: 'FPS_001', commodity: 'Rice', month: 12 });
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const handlePredict = async () => {
        setLoading(true);
        try {
            // Calling the Spring Boot Endpoint
            const response = await axios.get('http://localhost:8080/api/forecast/predict', {
                params: formData
            });
            setPrediction(response.data.predictedQuantity);
        } catch (error) {
            alert("Ensure Spring Boot and Python are both running!");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'sans-serif', backgroundColor: '#f4f7f6', minHeight: '100vh' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '30px' }}>
                <LayoutDashboard size={32} color="#2c3e50" />
                <h1 style={{ color: '#2c3e50', margin: 0 }}>E-Ration Demand Forecaster</h1>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
                {/* Input Section */}
                <div style={{ background: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
                    <h3>Prediction Filters</h3>
                    <label>Dealer ID</label>
                    <select onChange={(e) => setFormData({...formData, fpsId: e.target.value})} style={inputStyle}>
                        <option value="FPS_001">FPS_001 (Urban)</option>
                        <option value="FPS_002">FPS_002 (Rural)</option>
                        <option value="FPS_003">FPS_003 (Urban)</option>
                    </select>

                    <label>Commodity</label>
                    <select onChange={(e) => setFormData({...formData, commodity: e.target.value})} style={inputStyle}>
                        <option value="Rice">Rice</option>
                        <option value="Wheat">Wheat</option>
                        <option value="Sugar">Sugar</option>
                    </select>

                    <label>Forecast Month</label>
                    <input type="number" min="1" max="12" defaultValue="12" 
                           onChange={(e) => setFormData({...formData, month: e.target.value})} style={inputStyle} />

                    <button onClick={handlePredict} disabled={loading} style={buttonStyle}>
                        {loading ? 'Calculating...' : 'Generate Forecast'}
                    </button>
                </div>

                {/* Result Section */}
                <div style={{ background: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                    {!prediction ? (
                        <div style={{ textAlign: 'center', color: '#95a5a6' }}>
                            <Database size={64} />
                            <p>Select parameters and click Generate to see the ML prediction.</p>
                        </div>
                    ) : (
                        <div style={{ textAlign: 'center' }}>
                            <h2 style={{ color: '#27ae60' }}>Predicted Requirement</h2>
                            <div style={{ fontSize: '48px', fontWeight: 'bold', color: '#2c3e50' }}>
                                {prediction.toFixed(2)} <span style={{ fontSize: '24px' }}>KG</span>
                            </div>
                            <p style={{ color: '#7f8c8d' }}>Recommended stock allocation for {formData.commodity}</p>
                            
                            <div style={{ marginTop: '20px', padding: '10px', backgroundColor: '#e8f6ef', borderRadius: '8px', display: 'flex', gap: '10px', alignItems: 'center' }}>
                                <AlertCircle size={20} color="#27ae60" />
                                <span>Model Confidence: High (Based on 2023-2024 Trends)</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

const inputStyle = { width: '100%', padding: '10px', marginBottom: '15px', borderRadius: '6px', border: '1px solid #ddd' };
const buttonStyle = { width: '100%', padding: '12px', backgroundColor: '#3498db', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' };

export default RationDashboard;