# Model results

Data: data/processed/features.csv; 200 earthquake events; split by event_id (no event in both train and test).
Classifiers: recall target >= 90%. Thresholds are chosen on grouped out-of-fold predictions of the training events, then reported once on held-out events.

## Pipe burst / gas leak (`burst_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.340 | 0.912 | 0.860 | 0.885 | 0.661 | 0.902 | 0.853 |
| Random Forest | 0.526 | 0.912 | 0.858 | 0.885 | 0.651 | 0.900 | 0.849 |
| Hist Gradient Boosting | 0.360 | 0.902 | 0.859 | 0.880 | 0.656 | 0.900 | 0.851 |

## Fire (`fire_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.314 | 0.904 | 0.435 | 0.587 | 0.824 | 0.900 | 0.428 |
| Random Forest | 0.227 | 0.895 | 0.427 | 0.578 | 0.815 | 0.900 | 0.417 |
| Hist Gradient Boosting | 0.297 | 0.912 | 0.421 | 0.576 | 0.821 | 0.900 | 0.415 |

## Flood (`flood_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.315 | 0.917 | 0.333 | 0.489 | 0.816 | 0.900 | 0.315 |
| Random Forest | 0.176 | 0.912 | 0.315 | 0.469 | 0.801 | 0.900 | 0.304 |
| Hist Gradient Boosting | 0.292 | 0.920 | 0.321 | 0.476 | 0.809 | 0.900 | 0.304 |

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
