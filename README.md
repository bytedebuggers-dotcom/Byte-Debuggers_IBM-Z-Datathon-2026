# QuakeFix

**Multi-Hazard Urban Resilience & Emergency Orchestration Platform**
IBM Z Datathon 2026 | Challenge: Real-Time AI for Critical Decisions

Earthquakes rarely cause harm through shaking alone. Ruptured gas lines,
burst water mains, secondary flooding, collapsed roads and surges on
emergency medical services follow in a cascade, yet most emergency-response
software tracks each hazard in its own silo. QuakeFix unifies these domains
in a single decision-support dashboard so commanders don't have to connect
the dots by hand.

## What it does
- **Ingests data:** live seismic data from the USGS FDSN Event Web Service,
  road and building data from OpenStreetMap (OSMnx), and simulated
  utility-sensor telemetry.
- **Estimates risk:** XGBoost/LightGBM models estimate the probability of
  pipe bursts and gas leaks from shaking intensity.
- **Routes safely:** a NetworkX/OSMnx engine calculates safe paths for
  ambulances and evacuees around blocked or hazardous nodes.
- **Issues directives:** prescriptive actions such as isolating a valve
  sector or rerouting a unit, validated through FastAPI and Pydantic schemas.
- **Keeps humans in charge:** explicit Approve and Manual Override controls
  for every decision.
- **Works when networks fail:** low-bandwidth SMS/CSV export of priority actions.
- **Deploys securely:** containerised with Docker for data-sovereignty-
  respecting deployment on enterprise environments such as IBM LinuxONE.

## Important note
This prototype is built during the official event window (17-18 October
2026). Utility-failure models are trained on simulated, clearly labelled
synthetic data and serve as decision support, not as predictions of record.

## Tech stack
Python, Pandas, GeoPandas, XGBoost/LightGBM, NetworkX, OSMnx, FastAPI,
Pydantic, Streamlit, Folium/PyDeck, Docker

## Repository layout
`ml/` models and notebooks, `backend/` FastAPI service, `dashboard/`
Streamlit app, `models/` saved model files, `data/` data notes,
`docs/` documentation.
