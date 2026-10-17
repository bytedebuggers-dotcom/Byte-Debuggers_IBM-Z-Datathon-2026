import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

rows = []
for event_id in range(200):                      # Part A: 200 simulated earthquakes
    mag = rng.uniform(4.5, 7.5)
    for seg in range(100):                       # Part A: 100 map cells / segments each
        dist_km = rng.uniform(1, 80)
        shaking = max(0.0, mag * 1.5 - 1.2 * np.log1p(dist_km) + rng.normal(0, 0.4))
        pipe_age = int(rng.integers(5, 60))
        sensor_var = rng.uniform(0, 1)
        gas_main_dist_m = rng.uniform(5, 500)   # distance to nearest gas main
        elevation_m = rng.uniform(0, 60)        # low ground floods first
        building_age = int(rng.integers(5, 100))

        # RULE 1: pipe burst depends on shaking and pipe age
        p_burst = sigmoid(shaking - 4.5 + 0.03 * pipe_age + sensor_var - 1.5)
        burst = int(rng.random() < p_burst)

        # RULE 2: fire needs shaking plus a nearby gas main (more likely if the pipe burst)
        p_fire = sigmoid(-4.0 + 0.6 * shaking + 1.5 * burst
                         - 0.008 * gas_main_dist_m + 0.01 * building_age)
        fire = int(rng.random() < p_fire)

        # RULE 3: flood needs a burst or heavy shaking, and low ground
        p_flood = sigmoid(-3.0 + 0.4 * shaking + 1.5 * burst - 0.07 * elevation_m)
        flood = int(rng.random() < p_flood)

        # RULE 4: structural damage score (0 to 1) grows with shaking and building age
        damage = float(np.clip(0.10 * shaking + 0.004 * building_age
                               + rng.normal(0, 0.05), 0, 1))

        rows.append(dict(event_id=event_id, segment_id=seg, magnitude=mag,
                         dist_km=dist_km, shaking=shaking, pipe_age=pipe_age,
                         sensor_var=sensor_var, gas_main_dist_m=gas_main_dist_m,
                         elevation_m=elevation_m, building_age=building_age,
                         burst_48h=burst, fire_48h=fire, flood_48h=flood,
                         damage_score=damage, is_synthetic=True))

df = pd.DataFrame(rows)
df.to_csv("data/processed/features.csv", index=False)
print(df.shape)
print(df[["burst_48h", "fire_48h", "flood_48h", "damage_score"]].mean().round(3))