# Model results

Data: SYNTHETIC (built-in copy of the shared generator; hazard labels come from simple rules we wrote); 40 earthquake events; split by event_id (no event in both train and test).
Classifiers: recall target >= 90%. Thresholds are chosen on grouped out-of-fold predictions of the training events, then reported once on held-out events.

## Pipe burst / gas leak (`burst_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.268 | 0.960 | 0.704 | 0.812 | 0.833 | 0.901 | 0.699 |
| Random Forest | 0.298 | 0.955 | 0.697 | 0.806 | 0.811 | 0.901 | 0.692 |
| Hist Gradient Boosting | 0.268 | 0.947 | 0.709 | 0.811 | 0.809 | 0.901 | 0.687 |

## Fire (`fire_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.320 | 0.902 | 0.426 | 0.579 | 0.852 | 0.901 | 0.458 |
| Random Forest | 0.216 | 0.957 | 0.395 | 0.559 | 0.839 | 0.903 | 0.420 |
| Hist Gradient Boosting | 0.184 | 0.920 | 0.395 | 0.552 | 0.844 | 0.901 | 0.423 |

## Flood (`flood_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.308 | 0.872 | 0.315 | 0.463 | 0.789 | 0.901 | 0.291 |
| Random Forest | 0.163 | 0.863 | 0.289 | 0.433 | 0.777 | 0.901 | 0.271 |
| Hist Gradient Boosting | 0.076 | 0.906 | 0.257 | 0.400 | 0.759 | 0.901 | 0.235 |

## Structural damage score (`damage_score`, 0 to 1)

| Model | Test MAE | Test R2 | CV MAE | CV R2 |
|---|---|---|---|---|
| Ridge regression **(chosen)** | 0.039 | 0.908 | 0.039 | 0.940 |
| Random Forest | 0.040 | 0.902 | 0.041 | 0.936 |
| Hist Gradient Boosting | 0.039 | 0.907 | 0.041 | 0.937 |

## Limitations
1. Trained on simulated data, not validated on real earthquakes, fires or floods.
2. Held-out recall varies from earthquake to earthquake, so 90% is a target, not a guarantee.
3. The threshold trades precision for recall: more false alarms, fewer missed hazards.
4. Evacuation routing cannot see live crowds or traffic.
5. Gas main distance, elevation and building age are partly simulated, and the fire, flood and damage labels come from rules we wrote.
