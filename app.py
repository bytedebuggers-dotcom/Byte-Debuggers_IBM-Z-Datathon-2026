import os
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="QuakeFix Emergency Command Center",
    page_icon="🚨",
    layout="wide",
)

# Custom CSS Injector for Decorative Design & Styling
st.markdown("""
<style>
    .main {
        background-color: #F4EBD9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 6px 6px 0px 0px;
        color: #0B2A6B;
        font-weight: 600;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1FA37A !important;
        color: #FFFFFF !important;
    }
    div.stButton > button {
        background-color: #1FA37A;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #15805e;
        color: white;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-left: 5px solid #1FA37A;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .hero-banner {
        background: linear-gradient(90deg, #0B2A6B 0%, #1FA37A 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .insight-card {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0d8c5;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

# Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <h1 style="margin:0; font-size: 28px;">🚨 QuakeFix: Multi-Hazard Command Center</h1>
    <p style="margin:5px 0 0 0; font-size: 14px; opacity: 0.9;">AI-Powered Real-Time Seismic Risk Assessment & Street-Grid Aligned Emergency Routing</p>
</div>
""", unsafe_allow_html=True)

def api_get(path):
    try:
        r = requests.get(API + path, timeout=5)
        r.raise_for_status()
        return r
    except Exception as e:
        return None

if "prediction_data" not in st.session_state:
    initial_res = api_get("/events/latest")
    if initial_res:
        st.session_state["prediction_data"] = initial_res.json()
    else:
        st.session_state["prediction_data"] = {
            "event": {"id": "default", "magnitude": 6.1, "depth_km": 10.0, "lat": 37.7749, "lon": -122.4194},
            "directives": [], "zones": [], "routes": [],
            "burst_prob": 0.25, "fire_prob": 0.18, "flood_prob": 0.12, "damage_score": 3.4
        }

data = st.session_state["prediction_data"]
event = data["event"]

# Sidebar Decorative Panel
st.sidebar.markdown("### 🎛️ Tactical Control Center")
health_check = api_get("/health")
if health_check:
    st.sidebar.success("🟢 Backend API: Online (LinuxONE)")
else:
    st.sidebar.error("🔴 Backend API: Offline")

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Map Infrastructure Layers")
show_hazards = st.sidebar.checkbox("Hazard Zones", True)
show_hospitals = st.sidebar.checkbox("Hospitals", True)
show_assembly = st.sidebar.checkbox("Safe Assembly Points", True)
show_amb = st.sidebar.checkbox("Street-Aligned Ambulance Routes", True)
show_evac = st.sidebar.checkbox("Street-Aligned Evacuation Corridors", True)

tab_pred, tab_map, tab_queue, tab_export, tab_insights = st.tabs([
    "⚡ Live Risk Prediction", "🗺️ Command Map", "📋 Directive Queue", "📤 Dispatches", "📊 Model Insights"
])

with tab_pred:
    st.subheader("⚡ Real-Time Infrastructure Telemetry & Sliders")
    st.markdown("Adjust telemetry sliders below. Metrics, hazard positioning, and safe evacuation corridors recalculate **instantly**.")
    
    col1, col2 = st.columns(2)
    with col1:
        magnitude = st.slider("Earthquake Magnitude", 1.0, 9.0, float(event.get("magnitude", 6.1)), 0.1)
        st.caption(f"Current Value: **{magnitude:.1f}**")
        
        dist_km = st.slider("Epicentral Distance (km)", 0.1, 100.0, 12.0, 0.5)
        st.caption(f"Current Distance: **{dist_km:.1f} km**")
        
        shaking = st.slider("Shaking Intensity (MMI)", 0.0, 10.0, 5.2, 0.1)
        st.caption(f"Shaking Intensity Ratio: **{(shaking / 10.0) * 100:.1f}%**")
        
        pipe_age = st.slider("Pipeline Age (Years)", 1.0, 100.0, 40.0, 1.0)
        st.caption(f"Pipeline Lifespan Exhaustion: **{(pipe_age / 100.0) * 100:.1f}%**")

    with col2:
        sensor_var = st.slider("Sensor Variance", 0.0, 10.0, 0.6, 0.1)
        st.caption(f"Sensor Error Variance: **{(sensor_var / 10.0) * 100:.1f}%**")
        
        gas_main_dist_m = st.slider("Distance to Gas Main (m)", 0.0, 500.0, 60.0, 5.0)
        st.caption(f"Gas Proximity Risk Index: **{(gas_main_dist_m / 500.0) * 100:.1f}%**")
        
        elevation_m = st.slider("Elevation (Meters)", 0.0, 2000.0, 8.0, 1.0)
        st.caption(f"Elevation Profile Ratio: **{(elevation_m / 2000.0) * 100:.2f}%**")
        
        building_age = st.slider("Building Age (Years)", 1.0, 100.0, 70.0, 1.0)
        st.caption(f"Structural Degradation Level: **{(building_age / 100.0) * 100:.1f}%**")

    payload = {
        "features": [magnitude, dist_km, shaking, pipe_age, sensor_var, gas_main_dist_m, elevation_m, building_age]
    }
    try:
        res = requests.post(f"{API}/predict", json=payload, timeout=3)
        if res.status_code == 200:
            st.session_state["prediction_data"] = res.json()
            data = st.session_state["prediction_data"]
            event = data["event"]
    except Exception:
        pass

    st.markdown("### 📈 Current Calculated Assessment Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><h4>Pipe Burst Risk</h4><h2>{data.get("burst_prob", 0.25)*100:.1f}%</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h4>Fire Risk</h4><h2>{data.get("fire_prob", 0.18)*100:.1f}%</h2></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h4>Flood Risk</h4><h2>{data.get("flood_prob", 0.12)*100:.1f}%</h2></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><h4>Structural Damage</h4><h2>{data.get("damage_score", 3.4):.1f} / 10</h2></div>', unsafe_allow_html=True)

with tab_map:
    st.subheader(f"🗺️ San Francisco Tactical Grid | Magnitude {event['magnitude']} | Depth {event['depth_km']} km")
    
    all_zones = data.get("zones", [])
    
    hazard_zones = [z for z in all_zones if z.get("type") not in ["Hospital", "Assembly Point"] and show_hazards]
    hospital_zones = [z for z in all_zones if z.get("type") == "Hospital" and show_hospitals]
    assembly_zones = [z for z in all_zones if z.get("type") == "Assembly Point" and show_assembly]
    
    backend_routes = [rt for rt in data.get("routes", [])
                      if (rt["kind"] == "ambulance" and show_amb) or (rt["kind"] == "evacuation" and show_evac)]

    layers = []
    if hazard_zones:
        layers.append(pdk.Layer("ScatterplotLayer", hazard_zones, get_position=["lon", "lat"], get_fill_color="color", get_radius="radius", pickable=True, opacity=0.85))
    if hospital_zones:
        layers.append(pdk.Layer("ScatterplotLayer", hospital_zones, get_position=["lon", "lat"], get_fill_color="color", get_radius="radius", pickable=True, opacity=1.0))
    if assembly_zones:
        layers.append(pdk.Layer("ScatterplotLayer", assembly_zones, get_position=["lon", "lat"], get_fill_color="color", get_radius="radius", pickable=True, opacity=1.0))
    if backend_routes:
        # Skinnier, thin lines matching the street grid vectors cleanly
        layers.append(pdk.Layer("PathLayer", backend_routes, get_path="path", get_color="color", width_scale=10, width_min_pixels=2, pickable=True))

    view_state = pdk.ViewState(latitude=event["lat"], longitude=event["lon"], zoom=12, pitch=30)

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip={  # type: ignore
            "html": "<b>Location:</b> {name}<br/><b>Category:</b> {type}",
            "style": {"backgroundColor": "#0B2A6B", "color": "white"},
        },
    ))
    st.info("💡 **Routing Note:** Emergency vector paths are styled thin and programmatically aligned with San Francisco street-grid corridors (Van Ness & Mission axes).")

with tab_queue:
    st.subheader("📋 Emergency Directive Action Queue")
    st.markdown("Review AI-generated hazard responses, record commander notes, and execute **Approvals** or **Manual Overrides**.")
    
    for d in data.get("directives", []):
        with st.container():
            c1, c2, c3 = st.columns([4, 2, 4])
            c1.markdown(f"**{d['id']} (Priority {d['priority']})**\n\n*Hazard:* `{d['hazard']}`\n\n**Action:** {d['action']}")
            
            risk = d.get("risk")
            conf = d.get("confidence")
            c2.markdown(f"**Risk:** {risk:.0%}" if risk is not None else "**Risk:** N/A")
            c2.markdown(f"**Conf:** {conf:.0%}" if conf is not None else "")
            status_color = "orange" if d['status'] == "pending" else ("green" if d['status'] == "approved" else "red")
            c2.markdown(f"**Status:** :{status_color}[{d['status'].upper()}]")
            
            note = c3.text_input("Commander Note / Reason", key=f"note-{d['id']}")
            b1, b2 = c3.columns(2)
            if b1.button("✅ Approve", key=f"ok-{d['id']}", use_container_width=True):
                try:
                    resp = requests.post(f"{API}/directives/{d['id']}/decision", json={"decision": "approve", "note": note}, timeout=5)
                    if resp.status_code == 200:
                        st.success(f"Directive {d['id']} Approved.")
                        st.rerun()
                except Exception as ex:
                    st.error(f"Failed: {ex}")
            if b2.button("⚠️ Override", key=f"ov-{d['id']}", use_container_width=True):
                try:
                    resp = requests.post(f"{API}/directives/{d['id']}/decision", json={"decision": "override", "note": note}, timeout=5)
                    if resp.status_code == 200:
                        st.warning(f"Directive {d['id']} Overridden.")
                        st.rerun()
                except Exception as ex:
                    st.error(f"Failed: {ex}")
            st.divider()

with tab_export:
    st.subheader("📤 Emergency Dispatches & Audit Exports")
    sms_res = api_get("/export/sms")
    if sms_res is not None:
        st.code(sms_res.text, language="text")
        st.caption("Broadcast-ready text feed formatted for tactical first-responder radio relay and emergency SMS dispatchers.")
    
    st.markdown("---")
    csv_res = api_get("/export/csv")
    if csv_res is not None:
        st.download_button(
            label="📥 Download Action Log CSV",
            data=csv_res.text,
            file_name="quakefix_command_audit.csv",
            mime="text/csv",
            type="primary"
        )

with tab_insights:
    st.subheader("📊 Model Explainability & Performance Insights")
    st.markdown("Comprehensive evaluation and feature attribution across all **four core QuakeFix submodels** (Pipe Burst, Fire Risk, Flood Inundation, and Structural Damage)[cite: 2].")
    
    # Submodel 1 & 2 (Classification explainability plots)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("### 🚰 Submodel 1: Pipe Burst (Classifier)")
        st.markdown("Evaluates pipeline age, sensor error variance, and peak ground shaking intensity.")
        if os.path.exists("docs/explain_burst.png"):
            st.image("docs/explain_burst.png", caption="SHAP Global Feature Impact: Pipe Burst", use_container_width=True)
        else:
            st.info("`docs/explain_burst.png` pending generation.")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with c2:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 Submodel 2: Fire Risk (Classifier)")
        st.markdown("Assesses secondary ignition probability based on gas main proximity and building age.")
        if os.path.exists("docs/explain_fire.png"):
            st.image("docs/explain_fire.png", caption="SHAP Global Feature Impact: Fire Hazard", use_container_width=True)
        else:
            st.info("`docs/explain_fire.png` pending generation.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Submodel 3 & 4 (Flood and Structural Damage)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("### 🌊 Submodel 3: Flood Inundation (Classifier)")
        st.markdown("Calculates coastal/low-elevation vulnerability and burst water main runoff impact.")
        if os.path.exists("docs/explain_flood.png"):
            st.image("docs/explain_flood.png", caption="SHAP Global Feature Impact: Flood Risk", use_container_width=True)
        else:
            st.info("`docs/explain_flood.png` pending generation.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("### 🏗️ Submodel 4: Structural Damage (Regressor)")
        st.markdown("Estimates continuous structural degradation score (0 to 10 scale) using building age and MMI shaking.")
        if os.path.exists("docs/explain_damage.png"):
            st.image("docs/explain_damage.png", caption="Feature Attribution: Structural Collapse Score", use_container_width=True)
        else:
            st.markdown("""
            * **Model Architecture:** Random Forest Regressor[cite: 2]
            * **Target Metric:** Continuous Scale (0.0 - 10.0)[cite: 2]
            * **Primary Drivers:** Building Age, MMI Shaking Intensity, Elevation Profile
            * **Status:** Operational via `models/model.joblib`[cite: 2]
            """)
        st.markdown('</div>', unsafe_allow_html=True)