package com.eration.backend.service;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.util.HashMap;
import java.util.Map;

@Service
public class MLClient {
    private final RestTemplate restTemplate;
    private final String PYTHON_API_URL = "http://localhost:5000/predict";

    public MLClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public Double fetchPrediction(String fpsId, String commodity, int month) {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("fps_id", fpsId);
        requestBody.put("commodity", commodity);
        requestBody.put("month", month);
        requestBody.put("year", 2024);

        try {
            // Sending POST request to Flask
            Map<String, Object> response = restTemplate.postForObject(PYTHON_API_URL, requestBody, Map.class);
            return (Double) response.get("predicted_quantity");
        } catch (Exception e) {
            throw new RuntimeException("ML Service is down. Please run app.py first!");
        }
    }
}