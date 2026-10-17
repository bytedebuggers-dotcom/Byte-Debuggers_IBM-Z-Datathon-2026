# Model results

Data: SYNTHETIC (built-in generator; fire/flood/infrastructure labels are simulated); 40 earthquake events; split by event_id (no event in both train and test).
Recall target >= 90%. Thresholds are chosen on grouped out-of-fold predictions of the training events and reported once on held-out events.

## Pipeline burst (`burst_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.318 | 0.874 | 0.690 | 0.771 | 0.838 | 0.905 | 0.708 |
| Random Forest | 0.318 | 0.880 | 0.671 | 0.761 | 0.831 | 0.901 | 0.692 |
| Hist Gradient Boosting | 0.290 | 0.890 | 0.668 | 0.764 | 0.825 | 0.901 | 0.695 |

## Fire risk (`fire_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.306 | 0.874 | 0.332 | 0.482 | 0.796 | 0.901 | 0.351 |
| Random Forest | 0.157 | 0.921 | 0.265 | 0.411 | 0.770 | 0.901 | 0.318 |
| Hist Gradient Boosting | 0.175 | 0.945 | 0.270 | 0.420 | 0.760 | 0.901 | 0.313 |

## Flood risk (`flood_48h`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.377 | 0.874 | 0.231 | 0.365 | 0.675 | 0.907 | 0.232 |
| Random Forest | 0.177 | 0.882 | 0.219 | 0.351 | 0.619 | 0.902 | 0.224 |
| Hist Gradient Boosting | 0.189 | 0.908 | 0.217 | 0.350 | 0.606 | 0.902 | 0.225 |

## Infrastructure damage (`infra_damage`)

| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression **(chosen)** | 0.317 | 0.893 | 0.478 | 0.622 | 0.763 | 0.902 | 0.500 |
| Random Forest | 0.256 | 0.912 | 0.444 | 0.598 | 0.737 | 0.901 | 0.482 |
| Hist Gradient Boosting | 0.259 | 0.888 | 0.442 | 0.590 | 0.738 | 0.901 | 0.472 |

## Limitations
1. Trained on simulated data; not validated on real failures. Fire, flood and infrastructure labels come from simple documented rules, so those scores show the pipeline works, not real-world accuracy.
2. Held-out recall varies from earthquake to earthquake, so 90% is a target, not a guarantee.
3. The threshold trades precision for recall: more false alarms, fewer missed hazards.
4. The models use a small shared feature set (magnitude, distance, shaking, pipe age, sensor variance).
5. Advisory output only; a commander approves every action.
