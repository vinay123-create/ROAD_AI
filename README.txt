# RoadGuard AI — Machine Learning Module

## 1. Overview

RoadGuard AI uses a trained Random Forest classification model to predict road collision severity.

The model predicts one of three classes:

- **1 — Fatal**
- **2 — Serious**
- **3 — Slight**

## 2. Model Architecture

The ML module contains two main components:

- `roadguard_rf_model.pkl` — trained Random Forest classifier.
- `roadguard_preprocessor.pkl` — preprocessing transformer.

The preprocessor converts the 18 input features into 79 processed features before prediction.

## 3. Input Features

The model expects these 18 input features:

1. longitude
2. latitude
3. day_of_week
4. first_road_class
5. road_type
6. speed_limit
7. junction_detail
8. junction_control
9. second_road_class
10. pedestrian_crossing
11. light_conditions
12. weather_conditions
13. road_surface_conditions
14. urban_or_rural_area
15. trunk_road_flag
16. year
17. month
18. hour

The exact feature order and names are also available in `roadguard_features.json`.

## 4. Files

| File | Purpose |
|---|---|
| ROAD.ipynb | ML experiments and development |
| roadguard_rf_model.pkl | Trained classifier |
| roadguard_preprocessor.pkl | Input preprocessing |
| roadguard_bundle.pkl | Saved model bundle |
| roadguard_features.json | Input feature list |

## 5. Important Notes

- Input values must follow the dataset's feature definitions and coding conventions.
- The model predicts collision severity from recorded collision conditions. It is not a system for predicting whether an accident will happen.
- This is an experimental project model, not a validated emergency-response or safety-critical system.