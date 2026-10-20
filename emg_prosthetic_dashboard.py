import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

st.set_page_config(
    page_title="NeuroGrip | EMG Lab",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

GESTURES = {
    0: "Unmarked", 1: "Rest", 2: "Fist", 3: "Wrist Flexion",
    4: "Wrist Extension", 5: "Radial Deviation",
    6: "Ulnar Deviation", 7: "Extended Palm",
}
CHANNELS = [f"channel{i}" for i in range(1, 9)]
COLS = ["time"] + CHANNELS + ["class"]

# ---- Visual styling ----
st.markdown("""
<style>
:root { --accent: #5eead4; }
.stApp { background: radial-gradient(ellipse at 15% 0%, rgba(20,184,166,.10), transparent 34%), #0b1020; }
[data-testid="stHeader"] { background: rgba(11,16,32,.72); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -0.035em; }
h1 { font-weight: 800; }
[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(30,41,59,.86), rgba(15,23,42,.86)); border: 1px solid rgba(148,163,184,.18); padding: 16px 18px; border-radius: 16px; }
[data-testid="stMetricLabel"] { color: #a7b4c9; }
[data-testid="stMetricValue"] { color: #f8fafc; }
div[data-testid="stTabs"] button { border-radius: 10px 10px 0 0; font-weight: 650; }
div.stButton > button, div.stDownloadButton > button { border-radius: 10px; border: 1px solid rgba(94,234,212,.4); font-weight: 650; }
div.stButton > button[kind="primary"] { background: #14b8a6; color: #06111b; }
.panel { background: linear-gradient(145deg, rgba(30,41,59,.72), rgba(15,23,42,.76)); border: 1px solid rgba(148,163,184,.18); border-radius: 18px; padding: 18px 20px; }
.eyebrow { color: #5eead4; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 800; }
.subtle { color: #9aa9c0; }
.pill { display:inline-block; padding: 5px 10px; border-radius: 999px; background: rgba(45,212,191,.12); border:1px solid rgba(45,212,191,.28); color:#99f6e4; font-size:.78rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    raw = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", engine="python")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if not set(COLS).issubset(raw.columns):
        raw = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", header=None, names=COLS, engine="python")
    raw = raw[COLS]
    for c in COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=COLS).reset_index(drop=True)
    raw["class"] = raw["class"].astype(int)
    return raw

@st.cache_data(show_spinner=False)
def build_features(frame, window_size):
    xraw = frame[CHANNELS].to_numpy(dtype=np.float32, copy=False)
    labels = frame["class"].to_numpy(dtype=np.int16, copy=False)
    feature_rows, targets = [], []
    win = int(window_size)
    for start in range(0, len(frame) - win + 1, win):
        labs = labels[start:start + win]
        if labs[0] not in range(1, 8) or not np.all(labs == labs[0]):
            continue
        signal = xraw[start:start + win]
        row = {}
        for j, ch in enumerate(CHANNELS):
            v = signal[:, j]
            row[f"{ch}_MAV"] = np.mean(np.abs(v))
            row[f"{ch}_RMS"] = np.sqrt(np.mean(v * v))
            row[f"{ch}_VAR"] = np.var(v)
            row[f"{ch}_WL"] = np.abs(np.diff(v)).sum()
        feature_rows.append(row)
        targets.append(int(labs[0]))
    return pd.DataFrame(feature_rows), np.asarray(targets, dtype=int)

def pretty_class(c):
    return f"{int(c)} · {GESTURES.get(int(c), 'Unknown')}"

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="eyebrow">NEUROGRIP / CONTROL ROOM</div>', unsafe_allow_html=True)
    st.title("🦾 NeuroGrip")
    st.caption("EMG prosthetic-hand research dashboard")
    st.divider()
    uploaded = st.file_uploader("Load MYO EMG recording", type=["txt", "csv"], help="Expected columns: time, channel1–channel8, class.")
    st.markdown("**Analysis settings**")
    window_size = st.select_slider("Window size (samples)", options=[100, 150, 200, 250, 300, 400, 500], value=200)
    plot_points = st.select_slider("Plot detail", options=[1000, 2000, 3000, 5000, 8000], value=3000)
    st.divider()
    st.markdown('<span class="pill">OFFLINE DATA MODE</span>', unsafe_allow_html=True)
    st.caption("Recorded data analysis. Live MYO streaming and physical servo control are not connected.")

# ---- Hero ----
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="eyebrow">BIOSIGNAL ANALYTICS · PROTOTYPE</div>', unsafe_allow_html=True)
    st.title("EMG Prosthetic Hand Lab")
    st.markdown("Explore muscle activity, classify hand gestures, and preview a gesture-to-action workflow.")
with right:
    st.markdown('<div class="panel"><div class="eyebrow">SYSTEM STATUS</div><h3 style="margin:.4rem 0;color:#5eead4">● Ready</h3><div class="subtle">Waiting for a recording</div></div>', unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div class="panel">
      <div class="eyebrow">START HERE</div>
      <h2>Bring your EMG recording into the lab</h2>
      <p class="subtle">Upload the MYO bracelet text file from the sidebar. The dashboard will summarize the recording, let you inspect all eight channels, and train a baseline gesture classifier.</p>
      <span class="pill">8 channels</span> &nbsp; <span class="pill">7 labeled classes</span> &nbsp; <span class="pill">Feature extraction</span> &nbsp; <span class="pill">Model evaluation</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    data = load_data(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read this file. Check that it has time, channel1–channel8, and class columns. Details: {exc}")
    st.stop()

valid = data[data["class"].between(1, 7)]
n_classes = int(valid["class"].nunique())
duration_ms = float(data["time"].max() - data["time"].min()) if len(data) else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total samples", f"{len(data):,}")
k2.metric("Recording span", f"{duration_ms/1000:.1f} s")
k3.metric("Gesture classes", f"{n_classes}/7")
k4.metric("EMG channels", "8")
st.caption(f"Loaded file: **{uploaded.name}** · {len(data):,} valid rows · class 0 is excluded from model training.")

tab_overview, tab_signals, tab_model, tab_sim = st.tabs([
    "◈  Overview", "〰  Signal explorer", "⌁  Gesture AI", "✳  Hand simulator"
])

with tab_overview:
    a, b = st.columns([1.15, 1])
    with a:
        st.markdown("### Gesture distribution")
        counts = data["class"].value_counts().sort_index()
        count_frame = pd.DataFrame({
            "Gesture": [pretty_class(i) for i in counts.index],
            "Samples": counts.values
        }).set_index("Gesture")
        st.bar_chart(count_frame, color="#2dd4bf", height=300)
    with b:
        st.markdown("### Recording health")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write("**File check**")
        st.write("🟢 Numeric columns parsed" if data[COLS].notna().all().all() else "🟠 Some invalid values were removed")
        st.write(f"🟢 {len(CHANNELS)} signal channels detected")
        st.write(f"🟢 Labels found: {', '.join(map(str, sorted(data['class'].unique())))}")
        st.write("🟡 Class 0 is treated as unmarked data")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### Channel statistics")
        stats = data[CHANNELS].agg(["mean", "std", "min", "max"]).T
        st.dataframe(stats.style.format("{:.3e}"), use_container_width=True, height=240)
    with st.expander("Preview first 25 rows"):
        st.dataframe(data.head(25), use_container_width=True)

with tab_signals:
    st.markdown("### Inspect muscle activity")
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_channel = st.selectbox("EMG channel", CHANNELS)
        selected_class = st.selectbox("Gesture label", sorted(data["class"].unique()), format_func=pretty_class)
        st.caption("Display is downsampled for responsiveness. Feature extraction uses the original rows.")
    stride = max(1, len(data) // int(plot_points))
    p = data.iloc[::stride]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(p["time"], p[selected_channel], linewidth=.7, color="#2dd4bf")
    ax.set_title(f"{selected_channel} · complete recording")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    segment = data[data["class"] == selected_class].head(2500)
    if len(segment):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(segment["time"], segment[selected_channel], linewidth=.8, color="#a78bfa")
        ax.set_title(f"{selected_channel} · {GESTURES.get(int(selected_class))} (first {len(segment)} matching rows)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No samples for this label.")

with tab_model:
    st.markdown("### Train and evaluate a gesture classifier")
    st.markdown("Each non-overlapping window is summarized with **MAV, RMS, variance and waveform length** across all eight channels.")
    st.warning("The random split is a demonstration baseline. Nearby windows may be similar, so accuracy can be optimistic; test on separate recordings or subjects for a stronger result.")
    if st.button("Train Random Forest", type="primary", use_container_width=False):
        with st.spinner("Extracting features and training model..."):
            train_data = valid.reset_index(drop=True)
            X, y = build_features(train_data, int(window_size))
            time.sleep(.1)
        if len(X) < 10:
            st.error("Too few valid windows. Try a smaller window size or upload a longer recording.")
        elif len(np.unique(y)) < 2:
            st.error("At least two gesture classes are required.")
        elif pd.Series(y).value_counts().min() < 2:
            st.error("One or more classes have fewer than two windows. Try a smaller window size or use more recordings.")
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=120, max_depth=16, min_samples_leaf=2,
                                           class_weight="balanced_subsample", random_state=42, n_jobs=-1)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            st.session_state["ng_model"] = model
            st.session_state["ng_features"] = X.columns.tolist()
            st.session_state["ng_accuracy"] = accuracy_score(yte, pred)
            st.session_state["ng_yte"] = yte
            st.session_state["ng_pred"] = pred
            st.session_state["ng_X"] = X
            st.session_state["ng_y"] = y
            st.success(f"Training complete · {len(X):,} labeled windows analyzed")
    if "ng_model" in st.session_state:
        yte, pred = st.session_state["ng_yte"], st.session_state["ng_pred"]
        q1, q2, q3 = st.columns(3)
        q1.metric("Holdout accuracy", f"{st.session_state['ng_accuracy']*100:.1f}%")
        q2.metric("Test windows", f"{len(yte):,}")
        q3.metric("Model", "Random Forest")
        p1, p2 = st.columns([1, 1.2])
        labels = sorted(set(yte) | set(pred))
        with p1:
            st.markdown("#### Classification report")
            st.code(classification_report(yte, pred, labels=labels,
                target_names=[GESTURES.get(int(x), str(x)) for x in labels], zero_division=0))
        with p2:
            st.markdown("#### Confusion matrix")
            fig, ax = plt.subplots(figsize=(7, 5))
            ConfusionMatrixDisplay.from_predictions(yte, pred, labels=labels,
                display_labels=[GESTURES.get(int(x), str(x)) for x in labels],
                xticks_rotation=40, cmap="BuGn", ax=ax, colorbar=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        importances = pd.Series(st.session_state["ng_model"].feature_importances_,
                                index=st.session_state["ng_features"]).sort_values(ascending=False).head(15)
        st.markdown("#### Most informative features")
        st.bar_chart(importances.sort_values(), color="#a78bfa", horizontal=True)
        buf = io.BytesIO()
        joblib.dump({"model": st.session_state["ng_model"],
                     "feature_columns": st.session_state["ng_features"],
                     "gesture_names": GESTURES, "window_size": int(window_size),
                     "channels": CHANNELS}, buf)
        st.download_button("⬇ Download trained model", buf.getvalue(),
                           "neurogrip_emg_model.joblib", "application/octet-stream")
    else:
        st.info("Press **Train Random Forest** to produce model metrics and charts.")

with tab_sim:
    st.markdown("### Gesture-to-action preview")
    st.caption("This is a visual concept simulator only; it does not send commands to Arduino or move physical servos.")
    gesture = st.selectbox("Choose a gesture", list(range(1, 8)), format_func=pretty_class, key="sim_gesture")
    actions = {
        1: ("Relax fingers", "🖐️", "Neutral / rest"),
        2: ("Close fingers", "✊", "Fist"),
        3: ("Tilt wrist forward", "↘️", "Wrist flexion"),
        4: ("Tilt wrist backward", "↗️", "Wrist extension"),
        5: ("Tilt toward thumb side", "👉", "Radial deviation"),
        6: ("Tilt toward little-finger side", "👈", "Ulnar deviation"),
        7: ("Open fingers", "🖐️", "Extended palm"),
    }
    action, icon, desc = actions[gesture]
    s1, s2 = st.columns([1, 2])
    with s1:
        st.markdown(f'<div class="panel" style="text-align:center"><div style="font-size:5rem">{icon}</div><div class="eyebrow">SELECTED GESTURE</div><h2>{desc}</h2><span class="pill">Class {gesture}</span></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Proposed prosthetic action")
        st.markdown(f"## {action}")
        st.write("Suggested controller logic:")
        st.code(f"gesture_class = {gesture}\\n# map class {gesture} to a bounded, smoothed hand command")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("For real hardware, add confidence thresholds, prediction smoothing, a safe rest state, mechanical limits, and a separate regulated servo supply before testing.")

st.divider()
st.markdown('<div class="subtle">NEUROGRIP · Student research prototype · Recorded EMG analytics, not a medical device</div>', unsafe_allow_html=True)
