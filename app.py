# =============================================================================
#  Centrifugal Compressor — Lube Oil Pressure Dashboard
#  Run:  streamlit run app.py
# =============================================================================

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Optional TensorFlow (LSTM page is disabled if missing) ──────────────────
try:
    from tensorflow.keras.models import load_model as keras_load_model
    TF_OK = True
except ImportError:
    TF_OK = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Compressor Lube Oil Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAFETY_THRESHOLD = 1.8
TARGET           = "75PI823.pv"

SENSOR_CATEGORIES = {
    "Axial Displacement": ["75ZI800BA.pv","75ZI800BB.pv","75ZI801BA.pv","75ZI801BB.pv"],
    "Pressure"          : ["75PI808.pv","75PI823.pv","75PI870.pv","75PI845.pv","75PDI853.pv"],
    "Temperature"       : ["75TI821.pv","75TI822.pv","75TI827.pv","75TI828.pv",
                           "75TI824.pv","75TI829.pv","75TI830.pv","75TI831.pv",
                           "75TI832.pv","75TI836.pv","75TI834.pv"],
    "Vibration"         : ["75XI821BX.pv","75XI822BY.pv","75XI823BX.pv","75XI824BX.pv"],
    "Speed"             : ["75SI865R.pv"],
}

BASE = os.path.dirname(os.path.abspath(__file__))
def apath(fname): return os.path.join(BASE, fname)

# ─── Loaders ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    scaler = joblib.load(apath("scaler.pkl"))
    rf     = joblib.load(apath("rf_model.pkl"))
    from xgboost import XGBRegressor
    xgb_m = XGBRegressor()
    xgb_m.load_model(apath("xgb_model.json"))
    lstm_m = None
    if TF_OK and os.path.exists(apath("lstm_model.keras")):
        try:
            lstm_m = keras_load_model(apath("lstm_model.keras"))
        except Exception:
            lstm_m = None
    return scaler, rf, xgb_m, lstm_m

@st.cache_data(show_spinner="Loading data…")
def load_data():
    df = pd.read_feather(apath("df_clean.feather"))
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp").sort_index()
    metrics   = pd.read_csv(apath("metrics.csv"), index_col=0)
    X_shap    = pd.read_feather(apath("X_shap.feather"))
    shap_vals = np.load(apath("shap_vals.npy"))
    mean_shap = pd.read_csv(apath("mean_shap.csv"), index_col=0).squeeze()
    with open(apath("feature_cols.json")) as fh:
        feature_cols = json.load(fh)
    with open(apath("expected_value.json")) as fh:
        expected_value = float(json.load(fh)["expected_value"])
    return df, metrics, X_shap, shap_vals, mean_shap, feature_cols, expected_value

# ─── Feature engineering ─────────────────────────────────────────────────────
def build_features(df_input, feature_cols):
    df = df_input.copy()
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    for lag in [1, 2, 4, 8, 16]:
        df[f"{TARGET}_lag_{lag}"] = df[TARGET].shift(lag)
    for w in [4, 8, 16]:
        df[f"{TARGET}_rmean_{w}"] = df[TARGET].rolling(w).mean()
        df[f"{TARGET}_rstd_{w}"]  = df[TARGET].rolling(w).std()
    df["pressure_trend"]       = df[TARGET].diff()
    df["pressure_trend_4"]     = df[TARGET].diff(4)
    df["pressure_accel"]       = df["pressure_trend"].diff()
    df["pump_to_header_ratio"] = df["75PI808.pv"] / (df[TARGET] + 1e-6)
    tod = df.index.hour + df.index.minute / 60
    df["sin_hour"] = np.sin(2 * np.pi * tod / 24)
    df["cos_hour"] = np.cos(2 * np.pi * tod / 24)
    doy = df.index.dayofyear
    df["sin_doy"]  = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"]  = np.cos(2 * np.pi * doy / 365)
    df.dropna(inplace=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_cols]

# ─── Load everything ─────────────────────────────────────────────────────────
try:
    scaler, rf_model, xgb_model, lstm_model = load_models()
    df, metrics_df, X_shap, shap_vals, mean_shap, feature_cols, expected_value = load_data()
    assets_ok = True
except Exception as e:
    assets_ok  = False
    load_error = str(e)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Compressor Monitor")
    st.markdown("**Lube Oil Header Pressure**  \n`75PI823.pv`")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📊 Overview",
         "🤖 Model Performance",
         "🔍 SHAP Explainability",
         "🔮 Live Prediction",
         "📁 Data Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Safety threshold: **{SAFETY_THRESHOLD} kg/cm²**")
    if not TF_OK:
        st.warning("TensorFlow not found. LSTM unavailable.", icon="⚠️")
    if assets_ok:
        n_low = int((df[TARGET] < SAFETY_THRESHOLD).sum())
        clr   = "red" if n_low > 0 else "green"
        st.markdown(
            f"Low-pressure events: "
            f"<span style='color:{clr};font-weight:bold'>{n_low}</span>",
            unsafe_allow_html=True,
        )

# ─── Guard ───────────────────────────────────────────────────────────────────
if not assets_ok:
    st.error("⚠️ Could not load artefacts. Run the **Export Models** cell in the notebook first.")
    st.code(load_error)
    st.stop()

# =============================================================================
#  PAGE 1 — OVERVIEW
# =============================================================================
if page == "📊 Overview":
    st.title("📊 Compressor Overview — 2022")

    c1, c2, c3, c4 = st.columns(4)
    n_low = int((df[TARGET] < SAFETY_THRESHOLD).sum())
    c1.metric("Total Records",       f"{len(df):,}")
    c2.metric("Sensors",             df.shape[1])
    c3.metric("Low-Pressure Events", n_low,
              delta=f"{n_low/len(df)*100:.2f}% of records",
              delta_color="inverse")
    c4.metric("Mean Pressure", f"{df[TARGET].mean():.3f} kg/cm²")

    st.divider()

    st.subheader("Lube Oil Header Pressure — Full Year")
    mask = df[TARGET] < SAFETY_THRESHOLD
    fig  = go.Figure()
    fig.add_hrect(y0=0, y1=SAFETY_THRESHOLD, fillcolor="red", opacity=0.07,
                  annotation_text="⚠ DANGER ZONE", annotation_position="top left")
    fig.add_trace(go.Scattergl(x=df.index, y=df[TARGET], mode="lines",
                               name="Pressure", line=dict(color="steelblue", width=0.8)))
    fig.add_trace(go.Scattergl(x=df.index[mask], y=df[TARGET][mask], mode="markers",
                               name=f"< {SAFETY_THRESHOLD} kg/cm²",
                               marker=dict(color="crimson", size=3)))
    fig.update_layout(height=380, template="plotly_white",
                      xaxis_title="Date", yaxis_title="Pressure (kg/cm²)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean Pressure — Hour × Month")
    tmp = df[[TARGET]].copy()
    tmp["month"] = tmp.index.month
    tmp["hour"]  = tmp.index.hour
    pivot = tmp.pivot_table(index="hour", columns="month", values=TARGET, aggfunc="mean")
    pivot.columns = pd.to_datetime([f"2022-{m}-01" for m in pivot.columns]).strftime("%b")
    fig2 = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn",
                     labels=dict(x="Month", y="Hour of Day", color="kg/cm²"))
    fig2.update_layout(height=360, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sensor Distributions by Category")
    MAX_V = 2_000
    CAT_COLORS = ["#4472C4","#ED7D31","#A9D18E","#FF4444","#7030A0"]
    fig3 = make_subplots(rows=1, cols=5, subplot_titles=list(SENSOR_CATEGORIES.keys()))
    for ci, (cat, sensors) in enumerate(SENSOR_CATEGORIES.items(), 1):
        for s in sensors:
            if s not in df.columns:
                continue
            data = df[s].dropna()
            if len(data) > MAX_V:
                data = data.sample(MAX_V, random_state=42)
            fig3.add_trace(
                go.Violin(y=data, name=s, box_visible=True, meanline_visible=True,
                          fillcolor=CAT_COLORS[ci-1], opacity=0.6,
                          line_color="black", showlegend=False),
                row=1, col=ci,
            )
    fig3.update_layout(height=480, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
#  PAGE 2 — MODEL PERFORMANCE
# =============================================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    st.subheader("Test-Set Metrics (scaled space)")
    st.dataframe(
        metrics_df.style
        .format({"MAE": "{:.5f}", "RMSE": "{:.5f}", "R²": "{:.4f}"})
        .background_gradient(cmap="RdYlGn",   subset=["R²"])
        .background_gradient(cmap="RdYlGn_r", subset=["MAE","RMSE"]),
        use_container_width=True,
    )

    COLORS_M = ["#4472C4","#ED7D31","#70AD47"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["MAE ↓ better","RMSE ↓ better","R² ↑ better"])
    for ci, metric in enumerate(["MAE","RMSE","R²"], 1):
        fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[metric],
                             marker_color=COLORS_M, showlegend=False,
                             text=metrics_df[metric].round(4),
                             textposition="outside"),
                      row=1, col=ci)
    fig.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Radar — Normalised Scores (higher = better)")
    m = metrics_df.copy()
    for col in ["MAE","RMSE"]:
        m[col] = 1 - (m[col]-m[col].min()) / (m[col].max()-m[col].min()+1e-9)
    m["R²"] = (m["R²"]-m["R²"].min()) / (m["R²"].max()-m["R²"].min()+1e-9)
    cats = ["MAE score","RMSE score","R² score"]
    fig2 = go.Figure()
    for model in m.index:
        v = m.loc[model].tolist()
        fig2.add_trace(go.Scatterpolar(r=v+[v[0]], theta=cats+[cats[0]],
                                        fill="toself", name=model))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                       height=420, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
#  PAGE 3 — SHAP EXPLAINABILITY
# =============================================================================
elif page == "🔍 SHAP Explainability":
    st.title("🔍 SHAP Explainability — XGBoost")

    st.subheader("Global Feature Importance — Mean |SHAP|")
    fig = go.Figure(go.Bar(
        x=mean_shap.values[::-1], y=mean_shap.index[::-1],
        orientation="h", marker_color="#ED7D31",
    ))
    fig.update_layout(height=480, template="plotly_white",
                      xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Dependence Plot")
    top_feats  = list(mean_shap.head(15).index)
    selected   = st.selectbox("Feature", top_feats)
    color_feat = top_feats[1] if selected != top_feats[1] else top_feats[2]
    feat_idx   = list(X_shap.columns).index(selected)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=X_shap[selected], y=shap_vals[:, feat_idx],
        mode="markers",
        marker=dict(color=X_shap[color_feat], colorscale="RdYlBu_r",
                    colorbar=dict(title=color_feat), size=4, opacity=0.7),
        hovertemplate=(f"<b>{selected}</b>: %{{x:.3f}}<br>"
                       f"SHAP: %{{y:.4f}}<br>"
                       f"{color_feat}: %{{marker.color:.3f}}<extra></extra>"),
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(height=380, template="plotly_white",
                       xaxis_title=selected,
                       yaxis_title=f"SHAP value for {selected}")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Local Explanation — Waterfall")
    sample_idx = st.slider("Sample index (within SHAP subset)", 0, len(X_shap)-1, 0)
    base_val = expected_value

    sv        = shap_vals[sample_idx]
    top_idx   = np.argsort(np.abs(sv))[::-1][:12]
    top_names = [X_shap.columns[i] for i in top_idx]
    top_sv    = sv[top_idx]
    top_data  = X_shap.iloc[sample_idx].values[top_idx]
    cumsum    = np.cumsum(top_sv)
    bases     = [base_val] + list(base_val + cumsum[:-1])
    prediction = base_val + sv.sum()

    fig3 = go.Figure()
    for name, val, base, dval in zip(top_names, top_sv, bases, top_data):
        clr = "#e74c3c" if val > 0 else "#2ecc71"
        fig3.add_trace(go.Bar(
            x=[abs(val)], y=[f"{name} = {dval:.3f}"],
            orientation="h", marker_color=clr,
            base=min(base, base+val), showlegend=False,
            hovertemplate=(f"<b>{name}</b><br>SHAP: {val:+.4f}<br>"
                           f"Feature value: {dval:.4f}<extra></extra>"),
        ))
    fig3.add_vline(x=base_val,   line_dash="dot",  line_color="gray",
                   annotation_text=f"E[f(x)] = {base_val:.4f}")
    fig3.add_vline(x=prediction, line_dash="dash", line_color="navy",
                   annotation_text=f"f(x) = {prediction:.4f}")
    fig3.update_layout(height=440, template="plotly_white", barmode="overlay",
                       title=f"Sample {sample_idx} — Predicted = {prediction:.4f}",
                       xaxis_title="SHAP contribution (scaled pressure)")
    st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
#  PAGE 4 — LIVE PREDICTION
# =============================================================================
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction")
    st.info(
        "Upload a new Excel file with the **same sensor columns** as training data. "
        "Features are engineered automatically and the selected model is run."
    )

    model_options = ["XGBoost", "Random Forest"] + (["LSTM"] if lstm_model is not None else [])
    uploaded     = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    model_choice = st.selectbox("Model to use", model_options)

    if uploaded:
        with st.spinner("Reading file…"):
            df_new = pd.read_excel(uploaded)
            df_new["Timestamp"] = pd.to_datetime(df_new["Timestamp"])
            df_new = (df_new.sort_values("Timestamp")
                            .set_index("Timestamp")
                            .apply(pd.to_numeric, errors="coerce")
                            .ffill().bfill())

        st.success(f"Loaded {len(df_new):,} rows × {df_new.shape[1]} sensors")

        with st.spinner("Engineering features & scaling…"):
            X_new = build_features(df_new, feature_cols)
            all_cols = (list(scaler.feature_names_in_)
                        if hasattr(scaler, "feature_names_in_")
                        else feature_cols + [TARGET])
            df_scale = X_new.copy()
            if TARGET not in df_scale.columns:
                df_scale[TARGET] = 0.0
            for c in all_cols:
                if c not in df_scale.columns:
                    df_scale[c] = 0.0
            scaled = scaler.transform(df_scale[all_cols])
            df_sc  = pd.DataFrame(scaled, columns=all_cols, index=df_scale.index)
            X_pred = df_sc[feature_cols]

        with st.spinner(f"Running {model_choice}…"):
            SEQ_LEN = 8
            if model_choice == "XGBoost":
                y_pred = xgb_model.predict(X_pred)
            elif model_choice == "Random Forest":
                y_pred = rf_model.predict(X_pred)
            else:
                Xs = np.array([X_pred.values[i-SEQ_LEN:i]
                               for i in range(SEQ_LEN, len(X_pred))])
                y_pred = lstm_model.predict(Xs, verbose=0).flatten()
                X_pred = X_pred.iloc[SEQ_LEN:]

        fig = go.Figure()
        fig.add_hrect(y0=-999, y1=SAFETY_THRESHOLD, fillcolor="red", opacity=0.07,
                      annotation_text="⚠ DANGER ZONE", annotation_position="top left")
        if TARGET in df_new.columns:
            actual = df_new[TARGET].reindex(X_pred.index)
            fig.add_trace(go.Scatter(x=X_pred.index, y=actual, mode="lines",
                                     name="Actual",
                                     line=dict(color="steelblue", width=1)))
        fig.add_trace(go.Scatter(x=X_pred.index, y=y_pred, mode="lines",
                                 name=f"{model_choice} Prediction",
                                 line=dict(color="#ED7D31", width=1.5, dash="dash")))
        fig.update_layout(height=420, template="plotly_white",
                          title=f"{model_choice} — Predicted Lube Oil Pressure (scaled)",
                          xaxis_title="Timestamp", yaxis_title="Pressure (scaled)")
        st.plotly_chart(fig, use_container_width=True)

        alert_mask = y_pred < SAFETY_THRESHOLD
        n_alerts   = int(alert_mask.sum())
        if n_alerts > 0:
            st.error(f"🚨 {n_alerts} predicted time-steps are BELOW the safety threshold!")
            st.dataframe(
                pd.DataFrame({"Timestamp": X_pred.index[alert_mask],
                              "Predicted (scaled)": y_pred[alert_mask]}),
                use_container_width=True,
            )
        else:
            st.success("✅ No predicted values below the safety threshold.")

        st.download_button(
            "⬇️ Download Predictions CSV",
            pd.DataFrame({"Predicted": y_pred}, index=X_pred.index).to_csv().encode(),
            file_name="predictions.csv",
            mime="text/csv",
        )

# =============================================================================
#  PAGE 5 — DATA EXPLORER
# =============================================================================
elif page == "📁 Data Explorer":
    st.title("📁 Data Explorer")

    min_d = df.index.min().date()
    max_d = df.index.max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("From", min_d, min_value=min_d, max_value=max_d)
    end   = c2.date_input("To",   max_d, min_value=min_d, max_value=max_d)

    df_view = df.loc[str(start):str(end)]

    all_sensors = [c for c in df.columns if ".pv" in c]
    sel_sensors = st.multiselect("Sensors to plot", all_sensors,
                                 default=[TARGET, "75PI808.pv"])

    if sel_sensors:
        fig = go.Figure()
        for s in sel_sensors:
            fig.add_trace(go.Scattergl(x=df_view.index, y=df_view[s],
                                       mode="lines", name=s))
        if TARGET in sel_sensors:
            fig.add_hline(y=SAFETY_THRESHOLD, line_dash="dash",
                          line_color="red", annotation_text="Safety limit")
        fig.update_layout(height=420, template="plotly_white",
                          xaxis_title="Timestamp")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Descriptive Statistics")
    if sel_sensors:
        st.dataframe(df_view[sel_sensors].describe().T.style.format("{:.4f}"),
                     use_container_width=True)

    with st.expander("Raw data (first 500 rows of selection)"):
        st.dataframe(df_view[sel_sensors].head(500) if sel_sensors else df_view.head(500),
                     use_container_width=True)
