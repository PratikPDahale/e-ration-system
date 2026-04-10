package com.eration.backend.dto;

import lombok.Data;

@Data
public class ForecastRequest {
    private String fpsId;
    private String commodity;
    private int month;
    private int year;
}