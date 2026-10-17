# QuakeFix Data Dictionary

## Overview

This document outlines the dataset features used for training emergency response models in QuakeFix. The dataset combines real-world USGS event locations, OpenStreetMap infrastructure layouts, and physical/engineering simulation rules.

## Feature Definitions

| Column Name       | Description                                           | Unit                   | Source / Origin                |
| :---------------- | :---------------------------------------------------- | :--------------------- | :----------------------------- |
| `event_id`        | Unique identifier for the simulated earthquake event  | Index                  | Synthetic                      |
| `segment_id`      | Unique identifier for the geographic map cell/segment | Index                  | Synthetic                      |
| `magnitude`       | Earthquake moment magnitude ($M_w$)                   | Magnitude              | USGS / Synthetic               |
| `dist_km`         | Epicentral distance to the segment                    | Kilometers             | Synthetic                      |
| `shaking`         | Calculated ground motion intensity (PGA/MMI proxy)    | Acceleration/Intensity | Synthetic Model                |
| `pgv_cm_s`        | Peak Ground Velocity derived from shaking intensity   | cm/s                   | ALA / Hazus Derivation         |
| `length_km`       | Length of the utility pipe segment                    | Kilometers             | OSM / Synthetic                |
| `pipe_age`        | Age of the buried water/gas pipeline infrastructure   | Years                  | Municipal Records / Synthetic  |
| `sensor_var`      | Telemetry sensor variance/noise reading               | Unitless [0, 1]        | IoT Sensor Simulation          |
| `gas_main_dist_m` | Distance to the nearest high-pressure gas main        | Meters                 | OSM Infrastructure             |
| `elevation_m`     | Ground elevation above sea level                      | Meters                 | DEM / GIS Data                 |
| `building_age`    | Average age of structures in the segment              | Years                  | Municipal Data / Synthetic     |
| `burst_48h`       | Target: Pipeline burst probability within 48 hours    | Binary [0, 1]          | **FEMA Hazus / ALA Formula**   |
| `fire_48h`        | Target: Secondary fire outbreak within 48 hours       | Binary [0, 1]          | Empirical Simulation Rule      |
| `flood_48h`       | Target: Localized flooding risk within 48 hours       | Binary [0, 1]          | Empirical Simulation Rule      |
| `damage_score`    | Target: Continuous structural damage severity score   | Float [0, 1]           | Structural Vulnerability Model |
| `is_synthetic`    | Flag indicating whether the record is generated       | Boolean                | Metadata                       |

## Scientific Reference (Part B)

Pipeline burst labels (`burst_48h`) are calculated using published seismic fragility relationships from the **American Lifelines Alliance (ALA)** and the **FEMA Hazus Earthquake Technical Manual**.

The repair rate per kilometer is modeled as:
$$\text{Repair Rate} = K \times \text{PGV}$$
Where $K$ is the material/joint factor (set to $0.02$ for welded steel infrastructure) and $\text{PGV}$ is Peak Ground Velocity. The probability of segment failure is determined via the Poisson spatial distribution:
$$P(\text{burst}) = 1 - e^{-\text{Repair Rate} \times \text{length\_km}}$$
