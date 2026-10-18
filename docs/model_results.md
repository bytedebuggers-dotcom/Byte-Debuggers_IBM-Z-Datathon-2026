# Model results

Data: data/processed/features.csv; 200 earthquake events; split by event_id (no event in both train and test).
Classifiers: recall target >= 93%. Thresholds are chosen on grouped out-of-fold predictions of the training events, then reported once on held-out events.

## Pipe burst / gas leak (`burst_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.320 | 0.935 | 0.855 | 0.893 | 0.661 | 0.930 | 0.847 |
| Random Forest | 0.492 | 0.937 | 0.854 | 0.893 | 0.651 | 0.931 | 0.843 |
| Hist Gradient Boosting | 0.337 | 0.923 | 0.854 | 0.887 | 0.656 | 0.932 | 0.846 |

## Fire (`fire_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.261 | 0.927 | 0.408 | 0.566 | 0.824 | 0.930 | 0.402 |
| Random Forest | 0.189 | 0.928 | 0.401 | 0.560 | 0.815 | 0.930 | 0.393 |
| Hist Gradient Boosting | 0.246 | 0.938 | 0.399 | 0.560 | 0.821 | 0.931 | 0.392 |

## Flood (`flood_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.254 | 0.940 | 0.305 | 0.461 | 0.816 | 0.930 | 0.291 |
| Random Forest | 0.139 | 0.939 | 0.291 | 0.444 | 0.801 | 0.931 | 0.278 |
| Hist Gradient Boosting | 0.237 | 0.950 | 0.293 | 0.448 | 0.809 | 0.930 | 0.280 |

## Structural damage score (`damage_score`, 0 to 1)

| Model | Test MAE | Test R2 | CV MAE | CV R2 |
|---|---|---|---|---|
| Ridge regression **(chosen)** | 0.038 | 0.938 | 0.038 | 0.939 |
| Random Forest | 0.039 | 0.937 | 0.039 | 0.939 |
| Hist Gradient Boosting | 0.038 | 0.940 | 0.038 | 0.941 |

## Limitations
1. Trained on simulated data, not validated on real earthquakes, fires or floods.
2. Held-out recall varies from earthquake to earthquake, so 90% is a target, not a guarantee.
3. The threshold trades precision for recall: more false alarms, fewer missed hazards.
4. Evacuation routing cannot see live crowds or traffic.
5. Gas main distance, elevation and building age are partly simulated, and the fire, flood and damage labels come from rules we wrote.
