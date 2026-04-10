package com.eration.backend.controller;

import com.eration.backend.service.MLClient;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/forecast")
@CrossOrigin(origins = "http://localhost:3000") // Connects to React
public class ForecastController {

    private final MLClient mlClient;

    public ForecastController(MLClient mlClient) {
        this.mlClient = mlClient;
    }

    @GetMapping("/predict")
    public Map<String, Object> getPrediction(
            @RequestParam String fpsId,
            @RequestParam String commodity,
            @RequestParam int month) {

        Double prediction = mlClient.fetchPrediction(fpsId, commodity, month);

        return Map.of(
                "fpsId", fpsId,
                "commodity", commodity,
                "predictedQuantity", prediction,
                "status", "Success"
        );
    }
}