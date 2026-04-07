"""
Airlytics — AI-Powered Air Quality Intelligence Dashboard
=========================================================

A Streamlit application that turns raw air-quality telemetry into decisions.

Pipeline
--------
1. **Ingest**   Live + historical air-quality and meteorological telemetry from
                the Open-Meteo APIs (no API key required).
2. **Engineer** Resample to daily cadence, back-fill lags, derive AQI.
3. **Model**    * Regression  — multivariate Polynomial Ridge regression
                * Anomaly     — Isolation Forest spike detection
                * Survival    — Cox Proportional Hazards disaster-risk model
4. **Act**      Optional mirror to AWS S3 / SNS and Google Drive, plus an
                SMTP hazard alert. Every cloud integration degrades gracefully
                when credentials are absent, so the app always runs.

Secrets are read from environment variables / a local ``.env`` file — never
hard-code credentials in source control.

Run: ``streamlit run app.py``
"""

from __future__ import annotations

import os
import smtplib
import tempfile
from datetime import datetime, timedelta
from email.message import EmailMessage

import boto3
import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from streamlit_folium import st_folium

load_dotenv()

# NOTE: this must remain the very first Streamlit command in the script.
st.set_page_config(page_title="Airlytics", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------
# 1. CONFIGURATION (all secrets come from the environment)
# --------------------------------------------------------------------------
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", SMTP_USER)

GCP_KEY_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GCP_DRIVE_FOLDER_ID = os.getenv("GCP_DRIVE_FOLDER_ID", "")

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "")

AQ_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_API = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "Airlytics/1.0 (final-year project)"
REQUEST_TIMEOUT = 30

POLLUTANT_VARS = (
    "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
)
METEO_VARS = "temperature_2m,relative_humidity_2m"

# Open-Meteo field -> Airlytics column
COLUMN_MAP = {
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "ozone": "O3",
    "pm2_5": "PM25",
    "pm10": "PM10",
    "us_aqi": "AQI",
    "temperature_2m": "TEMP",
    "relative_humidity_2m": "HUMIDITY",
}

MODEL_FEATURES = ["CO", "NO2", "SO2", "O3", "TEMP", "HUMIDITY", "AQI_Lag1"]
FALLBACK_CSV = os.path.join("data", "latest_pollution_data.csv")


# --------------------------------------------------------------------------
# 2. STYLING
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ff4b4b;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #00ffcc !important; }
    .metric-label { font-size: 14px; color: #bbbbbb !important; text-transform: uppercase; }
    .stApp { background-color: #0e1117; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# 3. DATA ENGINE
# --------------------------------------------------------------------------
# US EPA breakpoints used to derive an AQI from a PM2.5 concentration (µg/m³).
PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(concentration: float) -> float:
    """Convert a PM2.5 concentration (µg/m³) to the corresponding US EPA AQI."""
    if concentration is None or np.isnan(concentration):
        return np.nan
    if concentration > 500.4:
        return 500.0
    for low_c, high_c, low_i, high_i in PM25_BREAKPOINTS:
        if low_c <= concentration <= high_c:
            return (high_i - low_i) / (high_c - low_c) * (concentration - low_c) + low_i
    return np.nan


def _get_json(url: str, params: dict) -> dict:
    """GET a JSON document, raising on any transport or HTTP failure."""
    response = requests.get(
        url,
        params={**params, "timezone": "auto"},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def _normalise(payload: dict) -> pd.DataFrame:
    """Flatten an Open-Meteo ``hourly`` payload into an Airlytics frame."""
    hourly = payload["hourly"]
    frame = pd.DataFrame({"DateTime": pd.to_datetime(hourly["time"])})
    for source, target in COLUMN_MAP.items():
        if source in hourly:
            frame[target] = pd.to_numeric(hourly[source], errors="coerce")
    return frame


def load_sample_data(days: int) -> pd.DataFrame:
    """Offline fallback so the dashboard still demos without a network."""
    frame = pd.read_csv(FALLBACK_CSV, parse_dates=["DateTime"])
    keep = ["DateTime", "CO", "NO2", "SO2", "O3", "PM25", "TEMP", "HUMIDITY", "AQI"]
    frame = frame[[c for c in keep if c in frame.columns]]
    for col in frame.columns.drop("DateTime"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.tail(days).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=900, max_entries=64)
def fetch_air_quality(latitude: float, longitude: float, days: int):
    """Fetch telemetry for a coordinate.

    Returns ``(DataFrame, source_label)``. Never raises — on failure it falls
    back to the bundled sample dataset so the app keeps working offline.
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    common = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }

    try:
        pollutants = _normalise(_get_json(AQ_API, {**common, "hourly": POLLUTANT_VARS}))
    except Exception:
        return load_sample_data(days), "📦 Bundled sample (API unreachable)"

    try:
        meteo = _normalise(_get_json(MET_API, {**common, "hourly": METEO_VARS}))
        pollutants = pollutants.merge(meteo, on="DateTime", how="left")
    except Exception:
        pass  # meteorology is optional; the model tolerates its absence

    if pollutants.empty:
        return load_sample_data(days), "📦 Bundled sample (no telemetry returned)"

    # Guard against sparse pollutant fields: any missing AQI is derived from PM2.5.
    if pollutants.get("AQI") is None or pollutants["AQI"].isna().all():
        pollutants["AQI"] = pollutants["PM25"].apply(pm25_to_aqi) if "PM25" in pollutants else np.nan

    # Collapse to a clean daily cadence — matches the "Historical Depth (Days)"
    # slider one-for-one and keeps the ML stage fast on long ranges.
    pollutants = pollutants.set_index("DateTime").resample("D").mean()
    pollutants = pollutants.dropna(how="all").ffill().bfill().reset_index()

    if "AQI" in pollutants:
        pollutants["AQI"] = pollutants["AQI"].fillna(pollutants["PM25"].apply(pm25_to_aqi))
    if "PM25" not in pollutants:
        pollutants["PM25"] = pollutants["AQI"]
    for column in pollutants.columns.drop("DateTime"):
        pollutants[column] = pollutants[column].fillna(pollutants[column].median())

    pollutants["AQI_Lag1"] = pollutants["AQI"].shift(1).bfill().fillna(pollutants["AQI"])
    return pollutants.dropna(subset=["AQI"]), "🟢 Live · Open-Meteo"


# --------------------------------------------------------------------------
# 4. CLOUD + ALERT INTEGRATIONS (all optional and failure-tolerant)
# --------------------------------------------------------------------------
def _aws_client(service: str):
    """Return a boto3 client, or ``None`` when credentials are unavailable."""
    try:
        session = boto3.Session(region_name=AWS_REGION)
        if session.get_credentials() is None:
            return None
        return session.client(service)
    except Exception:
        return None


def _write_export(frame: pd.DataFrame) -> str:
    """Materialise the current dataset as a temporary CSV for cloud upload."""
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="airlytics_", delete=False, encoding="utf-8"
    )
    with handle as fh:
        frame.to_csv(fh, index=False)
    return fh.name


def _export_and_upload(frame: pd.DataFrame, *actions) -> list[str]:
    """Write a temporary CSV, run each upload callable against it, then clean up."""
    path = _write_export(frame)
    try:
        return [action(path) for action in actions]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def upload_to_s3(file_path: str) -> str:
    """Upload a local CSV to S3. Returns a human-readable status."""
    if not S3_BUCKET_NAME:
        return "ℹ️ Set `S3_BUCKET_NAME` in `.env` to enable S3 sync."
    client = _aws_client("s3")
    if client is None:
        return "ℹ️ AWS credentials not found — configure `.env` or `~/.aws`."
    key = f"airlytics/{datetime.now():%Y%m%d_%H%M%S}/latest_pollution_data.csv"
    try:
        with open(file_path, "rb") as fh:
            body = fh.read()
        client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=body,
            ContentType="text/csv",
        )
        return f"✅ Synced to `s3://{S3_BUCKET_NAME}/{key}`"
    except (ClientError, BotoCoreError) as exc:
        return f"❌ S3 sync failed: {exc}"


def publish_sns_alert(subject: str, message: str) -> str:
    """Publish a hazard alert to an SNS topic."""
    if not SNS_TOPIC_ARN:
        return "ℹ️ Set `SNS_TOPIC_ARN` in `.env` to enable SNS alerts."
    client = _aws_client("sns")
    if client is None:
        return "ℹ️ AWS credentials not found — configure `.env` or `~/.aws`."
    try:
        client.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)
        return "✅ SNS hazard alert published."
    except (ClientError, BotoCoreError) as exc:
        return f"❌ SNS publish failed: {exc}"


def upload_to_google_drive(file_path: str, folder_id: str) -> str:
    """Mirror a CSV into a Google Drive folder using a service account."""
    if not GCP_KEY_FILE:
        return "ℹ️ Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to enable Drive mirror."
    if not folder_id:
        return "ℹ️ Set `GCP_DRIVE_FOLDER_ID` in `.env` to choose a Drive folder."
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        scopes = ["https://www.googleapis.com/auth/drive"]
        creds = service_account.Credentials.from_service_account_file(
            GCP_KEY_FILE, scopes=scopes
        )
        service = build("drive", "v3", credentials=creds)
        metadata = {
            "name": f"Pollution_Backup_{datetime.now():%Y%m%d_%H%M}.csv",
            "parents": [folder_id],
        }
        media = MediaFileUpload(file_path, mimetype="text/csv")
        service.files().create(body=metadata, media_body=media, fields="id").execute()
        return "✅ Mirrored to Google Drive."
    except Exception as exc:  # credentials, quota, network — surface it cleanly
        return f"❌ Drive mirror failed: {exc}"


def send_hazard_email(location: str, aqi: float, risk_pct: float, days: int, frame):
    """Send the hazard report over SMTP. Requires SMTP_USER + SMTP_APP_PASSWORD."""
    if not (SMTP_USER and SMTP_APP_PASSWORD and ALERT_RECIPIENT):
        return "ℹ️ Email not configured — set `SMTP_USER`, `SMTP_APP_PASSWORD` and `ALERT_RECIPIENT`."

    body = f"""
AIRLYTICS AUTOMATED HAZARD DISPATCH
-----------------------------------
LOCATION   : {location.upper()}
REPORT TIME: {datetime.now():%Y-%m-%d %H:%M:%S}

CRITICAL METRICS
- Current AQI        : {aqi:.1f}
- AI-Predicted Risk  : {risk_pct:.1f}% within {days} day(s)

AI ANALYSIS
The Cox Proportional Hazards model flagged a statistically significant
environmental hazard for this location.

RECOMMENDED ACTIONS
1. Issue a public health advisory for {location}.
2. Advise N95 masks for sensitive groups.
3. Restrict heavy vehicle movement in the central zone.

This is an automated message from the Airlytics AI Dashboard.
"""
    message = EmailMessage()
    message["Subject"] = f"⚠️ EMERGENCY: High Pollution Hazard Detected in {location}"
    message["From"] = f"Airlytics AI Monitor <{SMTP_USER}>"
    message["To"] = ALERT_RECIPIENT
    message.set_content(body)

    if frame is not None:
        message.add_attachment(
            frame.to_csv(index=False).encode("utf-8"),
            maintype="text",
            subtype="csv",
            filename=f"Airlytics_Backup_{location}.csv",
        )

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(SMTP_USER, SMTP_APP_PASSWORD)
            smtp.send_message(message)
        return f"✅ Hazard report dispatched to {ALERT_RECIPIENT}."
    except Exception as exc:
        return f"❌ Email dispatch failed: {exc}"


# --------------------------------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------------------------------
geolocator = Nominatim(user_agent="airlytics_dashboard_v2")

st.session_state.setdefault("coords", [28.6139, 77.2090])  # New Delhi
st.session_state.setdefault("scraped_df", None)
st.session_state.setdefault("map_search_sync", "Delhi")
st.session_state.setdefault("data_source", "Not loaded")

# Load a default location on first paint so the dashboard is never empty.
# This runs before the sidebar so the "Data source" caption is never stale.
if st.session_state["scraped_df"] is None:
    _lat, _lon = st.session_state["coords"]
    with st.spinner("Loading air-quality telemetry…"):
        _frame, _source = fetch_air_quality(_lat, _lon, 90)
    st.session_state["scraped_df"] = _frame
    st.session_state["data_source"] = _source

with st.sidebar:
    st.title("🚀 AIRLYTICS CONTROL")

    search_query = st.text_input(
        "🔍 Search Location", value=st.session_state["map_search_sync"]
    )
    hist_range = st.slider("Historical Depth (Days)", 30, 1500, 100, step=10)

    if st.button("Deep Extract & Analyze", use_container_width=True):
        try:
            place = geolocator.geocode(search_query, timeout=15)
        except Exception:
            place = None
        if place:
            st.session_state["coords"] = [place.latitude, place.longitude]
            frame, source = fetch_air_quality(place.latitude, place.longitude, hist_range)
            st.session_state["scraped_df"] = frame
            st.session_state["data_source"] = source
            st.session_state["map_search_sync"] = search_query
            st.rerun()
        else:
            st.error(f"Could not resolve location: {search_query}")

    st.caption(f"Data source: {st.session_state['data_source']}")

    st.divider()
    st.header("☁️ Cloud Services")

    col_s3, col_sns = st.columns(2)
    aws_upload = col_s3.button("📤 S3", use_container_width=True)
    aws_alert = col_sns.button("🚨 SNS", use_container_width=True)
    aws_drive = st.button("📁 Drive Mirror", use_container_width=True)
    send_email = st.button("📧 Send Hazard Email", use_container_width=True)

    with st.expander("⚙️ Cloud status"):
        st.write("S3 bucket:", S3_BUCKET_NAME or "not set")
        st.write("SNS topic:", SNS_TOPIC_ARN or "not set")
        st.write("Drive key:", "configured" if GCP_KEY_FILE else "not set")
        st.write("SMTP:", "configured" if SMTP_APP_PASSWORD else "not set")


# --------------------------------------------------------------------------
# 6. CLOUD BUTTON HANDLERS
# --------------------------------------------------------------------------
active_df = st.session_state["scraped_df"]
current_aqi = float(active_df["AQI"].iloc[-1]) if active_df is not None else 0.0
risk_pct = float(st.session_state.get("current_risk_score", 0.0))
forecast_days = int(st.session_state.get("f_days", 7))

if aws_upload:
    if active_df is None:
        st.sidebar.error("Run 'Deep Extract' first to generate data.")
    else:
        for status in _export_and_upload(
            active_df,
            upload_to_s3,
            lambda path: upload_to_google_drive(path, GCP_DRIVE_FOLDER_ID),
        ):
            st.sidebar.write(status)

if aws_alert:
    if active_df is None:
        st.sidebar.error("Run 'Deep Extract' first to generate data.")
    else:
        st.sidebar.info(f"Dispatching hazard alert for {search_query}…")
        subject = f"⚠️ EMERGENCY: High Pollution Hazard Detected in {search_query}"
        detail = (
            f"Location: {search_query}\nAQI: {current_aqi:.2f}\n"
            f"Risk: {risk_pct:.2f}% within {forecast_days} days\n"
            "Generated by the Airlytics dashboard."
        )
        st.sidebar.write(publish_sns_alert(subject, detail))

if aws_drive:
    if active_df is None:
        st.sidebar.error("Run 'Deep Extract' first to generate data.")
    else:
        for status in _export_and_upload(
            active_df, lambda path: upload_to_google_drive(path, GCP_DRIVE_FOLDER_ID)
        ):
            st.sidebar.write(status)

if send_email:
    if active_df is None:
        st.sidebar.error("Run 'Deep Extract' first to generate data.")
    else:
        st.sidebar.write(
            send_hazard_email(search_query, current_aqi, risk_pct, forecast_days, active_df)
        )


# --------------------------------------------------------------------------
# 7. MAIN DASHBOARD
# --------------------------------------------------------------------------
st.title("🛡️ Airlytics: Advanced Multivariate AI Dashboard")
st.caption(
    "Real-time telemetry · anomaly detection · survival-risk forecasting · multi-cloud sync"
)

c_map, c_vis = st.columns([1, 1.2])

with c_map:
    st.subheader("Geospatial Selection")
    m = folium.Map(location=st.session_state["coords"], zoom_start=9)
    folium.Circle(
        location=st.session_state["coords"],
        radius=10_000,
        color="#ff4b4b",
        fill=True,
        fill_opacity=0.2,
    ).add_to(m)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=400, width=550, key="main_map")

    if map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        st.session_state["coords"] = [lat, lng]
        try:
            reverse = geolocator.reverse(f"{lat}, {lng}", language="en", timeout=15)
            address = reverse.raw.get("address", {})
            st.session_state["map_search_sync"] = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or f"Area at {lat:.2f}, {lng:.2f}"
            )
        except Exception:
            st.session_state["map_search_sync"] = f"Area at {lat:.2f}, {lng:.2f}"

        with st.spinner("Loading telemetry for the selected point…"):
            frame, source = fetch_air_quality(lat, lng, hist_range)
        st.session_state["scraped_df"] = frame
        st.session_state["data_source"] = source
        st.rerun()

with c_vis:
    if st.session_state["scraped_df"] is not None:
        st.subheader("AQI Historical Trend")
        st.area_chart(
            st.session_state["scraped_df"].set_index("DateTime")["AQI"],
            color="#ff4b4b",
            x_label="Timeline",
            y_label="Air Quality Index (AQI)",
        )
    else:
        st.info("💡 Click the map or use the sidebar to load sensor telemetry.")


# --------------------------------------------------------------------------
# 8. AI RESEARCH LAB
# --------------------------------------------------------------------------
df = st.session_state["scraped_df"]

if df is None or len(df) < 5:
    st.info("At least 5 days of telemetry are required to run the models.")
    st.stop()

df = df.copy()
current_aqi = float(df["AQI"].iloc[-1])
features = [f for f in MODEL_FEATURES if f in df.columns]
X = df[features].fillna(0)
y = df["AQI"].fillna(0)

# Keep model complexity proportional to the sample size: a degree-4 surface is
# only identifiable once there are plenty of daily observations.
if len(df) >= 800:
    degree = 4
elif len(df) >= 250:
    degree = 3
else:
    degree = 2

reg = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=degree, include_bias=False),
    Ridge(alpha=10.0),
)
reg.fit(X, y)
df["Predicted_AQI"] = reg.predict(X)
df["Residuals"] = y - df["Predicted_AQI"]

mae = mean_absolute_error(y, df["Predicted_AQI"])
mse = mean_squared_error(y, df["Predicted_AQI"])
rmse = float(np.sqrt(mse))
r2 = r2_score(y, df["Predicted_AQI"])

tab_reg, tab_anom, tab_risk, tab_stat = st.tabs(
    ["📈 Regression", "🚨 Anomalies", "⏳ Survival", "📊 Data"]
)

# ---- Regression ----------------------------------------------------------
with tab_reg:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4], ["MAE", "MSE", "RMSE", "R-Squared"], [mae, mse, rmse, r2]
    ):
        col.markdown(
            f"""<div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val:.4f}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    st.write(f"**Actual vs. Predicted AQI — degree-{degree} Polynomial Ridge**")

    fig_reg, ax_reg = plt.subplots(figsize=(12, 5))
    ax_reg.plot(df["DateTime"].tail(100), df["AQI"].tail(100), label="Actual AQI",
                marker="o", markersize=4, alpha=0.7)
    ax_reg.plot(df["DateTime"].tail(100), df["Predicted_AQI"].tail(100),
                label="Predicted AQI", linestyle="--", color="#ff4b4b", linewidth=2)
    ax_reg.set_xlabel("Date")
    ax_reg.set_ylabel("Air Quality Index (AQI)")
    ax_reg.legend()
    fig_reg.autofmt_xdate()
    st.pyplot(fig_reg)

    st.write("**Residual Analysis**")
    fig_res, ax_res = plt.subplots(figsize=(10, 4))
    sns.scatterplot(x=df["Predicted_AQI"], y=df["Residuals"], color="#00ffcc",
                    alpha=0.5, ax=ax_res)
    ax_res.axhline(0, color="#ff4b4b", linestyle="--")
    ax_res.set_xlabel("Predicted AQI")
    ax_res.set_ylabel("Residual (Actual − Predicted)")
    st.pyplot(fig_res)

# ---- Anomalies -----------------------------------------------------------
with tab_anom:
    iso = IsolationForest(contamination=0.05, random_state=42).fit(df[["AQI"]])
    df["Anom_Score"] = iso.decision_function(df[["AQI"]])
    df["Status"] = np.where(iso.predict(df[["AQI"]]) == -1, "Anomaly", "Normal")
    anomalies = int((df["Status"] == "Anomaly").sum())
    st.metric("Detected Anomalies", anomalies, f"{anomalies / len(df):.1%} of samples")

    st.write("**Pollutant Spikes**")
    fig_anom, ax_anom = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=df, x="DateTime", y="AQI", hue="Status",
                    palette={"Normal": "#00ffcc", "Anomaly": "#ff4b4b"}, ax=ax_anom)
    ax_anom.set_xlabel("Date")
    ax_anom.set_ylabel("Air Quality Index (AQI)")
    st.pyplot(fig_anom)

    st.write("**Anomaly Score Distribution**")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Anom_Score"], kde=True, color="#00ffcc", ax=ax_dist)
    ax_dist.axvline(0, color="#ff4b4b", linestyle="--")
    st.pyplot(fig_dist)

# ---- Survival ------------------------------------------------------------
with tab_risk:
    df["Event"] = (df["AQI"] > (df["AQI"].mean() + df["AQI"].std())).astype(int)

    # Cox regression needs events to exist; fall back to the worst decile so a
    # near-constant series still produces a usable risk curve.
    if df["Event"].sum() < 2:
        df["Event"] = (df["AQI"] >= df["AQI"].quantile(0.85)).astype(int)

    df["Duration"] = range(1, len(df) + 1)
    cph = None

    if df["Event"].sum() >= 2:
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(
                df[["Duration", "Event", "AQI"]], "Duration", "Event"
            )
        except (ConvergenceError, ConvergenceWarning, ValueError) as exc:
            st.warning(f"Cox model did not converge: {exc}")

    f_days = st.slider("Forecast Risk Window (Days)", 1, 30, 7)
    st.session_state["f_days"] = f_days

    if cph is None:
        st.warning("Not enough variance in this dataset to fit the survival model.")
        risk = 0.0
    else:
        risk = float(
            1 - cph.predict_survival_function(df.tail(1), times=[f_days]).values[0][0]
        )

    risk_pct = max(0.0, min(100.0, risk * 100))
    st.session_state["current_risk_score"] = risk_pct

    gauge = (
        st.error if risk_pct > 70 else st.warning if risk_pct > 40 else st.success
    )
    gauge(f"⏳ Disaster Hazard Probability: {risk_pct:.2f}% in the next {f_days} day(s).")

    if cph is not None:
        fig_cox, ax_cox = plt.subplots(figsize=(10, 5))
        st.write("**Cox Survival Curves — probability of avoiding an AQI disaster**")
        cph.plot_partial_effects_on_outcome(
            covariates="AQI", values=[100, 200, 300], ax=ax_cox
        )
        ax_cox.set_xlabel("Time Duration (days)")
        ax_cox.set_ylabel("Survival Probability (0.0 – 1.0)")
        st.pyplot(fig_cox)

    # Two independent triggers: an unhealthy absolute AQI, or a high modelled
    # hazard probability. Report only the conditions that actually fired.
    triggered = []
    if current_aqi > 150:
        triggered.append(f"AQI {current_aqi:.0f} is in the unhealthy range")
    if risk_pct > 70:
        triggered.append(
            f"{risk_pct:.1f}% disaster probability within {f_days} day(s)"
        )

    if triggered:
        st.error("🚨 HAZARD ALERT — " + "; ".join(triggered) + ".")
        st.caption(
            "Alerts are **not** sent automatically. Use *📧 Send Hazard Email* or "
            "*🚨 SNS* in the sidebar to dispatch one."
        )
    else:
        st.success("✅ AI analysis: environment is currently stable.")

# ---- Data ----------------------------------------------------------------
with tab_stat:
    st.subheader("Multivariate Correlation Matrix")
    corr_cols = [c for c in features + ["AQI"] if c in df.columns]
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="viridis", ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)
    st.download_button(
        "📥 Export CSV",
        df.to_csv(index=False).encode("utf-8"),
        "airlytics_data.csv",
        "text/csv",
    )
