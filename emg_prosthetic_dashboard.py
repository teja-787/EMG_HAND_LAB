import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="EMG Prosthetic Hand Lab", page_icon="🦾", layout="wide")

GESTURE_NAMES = {
    0: "Unmarked", 1: "Rest", 2: "Fist", 3: "Wrist Flexion",
    4: "Wrist Extension", 5: "Radial Deviation",
    6: "Ulnar Deviation", 7: "Extended Palm"
}
CHANNELS = [f"channel{i}" for i in range(1, 9)]
COLS = ["time"] + CHANNELS + ["class"]

st.title("🦾 EMG Prosthetic Hand Lab")
st.caption("Analyze MYO bracelet EMG data, train a gesture classifier, and simulate prosthetic-hand actions.")

with st.sidebar:
    st.header("Project controls")
    uploaded = st.file_uploader("Upload raw EMG .txt file", type=["txt", "csv"])
    st.info("Expected columns: time, channel1–channel8, class")
    window_size = st.number_input("Window size (samples)", min_value=50, max_value=2000, value=200, step=50)
    test_size = st.slider("Test set proportion", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    st.caption("This dashboard uses recorded data; live bracelet streaming is not included.")

@st.cache_data(show_spinner="Loading EMG data...")
def load_data(file_bytes, filename):
    import io
    # Try headered data first; if columns do not match, retry as headerless.
    try:
        d = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", engine="python")
        d.columns = [str(c).strip().lower() for c in d.columns]
        if not set(COLS).issubset(d.columns):
            raise ValueError("No recognized header")
        d = d[COLS]
    except Exception:
        d = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", header=None, names=COLS, engine="python")
    for c in COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=COLS).reset_index(drop=True)
    d["class"] = d["class"].astype(int)
    return d

@st.cache_data(show_spinner="Extracting EMG features...")
def extract_features(data, window_size):
    rows, labels = [], []
    signals = data[CHANNELS].to_numpy(dtype=np.float32)
    classes = data["class"].to_numpy(dtype=np.int16)
    step = int(window_size)  # non-overlapping windows for speed
    for start in range(0, len(data) - int(window_size) + 1, step):
        end = start + int(window_size)
        labs = classes[start:end]
        if labs[0] not in range(1, 8) or not np.all(labs == labs[0]):
            continue
        w = signals[start:end]
        row = {}
        for i, ch in enumerate(CHANNELS):
            x = w[:, i]
            row[f"{ch}_MAV"] = float(np.mean(np.abs(x)))
            row[f"{ch}_RMS"] = float(np.sqrt(np.mean(x * x)))
            row[f"{ch}_VAR"] = float(np.var(x))
            row[f"{ch}_WL"] = float(np.sum(np.abs(np.diff(x))))
        rows.append(row)
        labels.append(int(labs[0]))
    return pd.DataFrame(rows), np.asarray(labels, dtype=int)

if uploaded is None:
    st.markdown("""
    ### Get started
    1. Upload your `raw_data_...txt` file in the sidebar.
    2. Explore the EMG channels and class distribution.
    3. Train the model and inspect its results.
    4. Use the simulator to preview a gesture-to-hand action mapping.
    """)
    st.stop()

try:
    data = load_data(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

# Header metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("EMG samples", f"{len(data):,}")
m2.metric("Channels", "8")
m3.metric("Gesture classes", str(data.loc[data["class"].between(1, 7), "class"].nunique()))
m4.metric("Missing values", f"{int(data[COLS].isna().sum().sum()):,}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "〰️ Signal explorer", "🧠 Model lab", "🦾 Hand simulator"])

with tab1:
    st.subheader("Gesture distribution")
    counts = data["class"].value_counts().sort_index()
    chart_df = pd.DataFrame({
        "Gesture": [f"{int(c)} — {GESTURE_NAMES.get(int(c), 'Unknown')}" for c in counts.index],
        "Samples": counts.values
    }).set_index("Gesture")
    st.bar_chart(chart_df)
    st.subheader("Data preview")
    st.dataframe(data.head(30), use_container_width=True)
    st.subheader("Channel summary")
    st.dataframe(data[CHANNELS].describe().T, use_container_width=True)

with tab2:
    st.subheader("Inspect an EMG channel")
    selected_channel = st.selectbox("Channel", CHANNELS)
    max_points = st.slider("Maximum plotted points", 500, 10000, 3000, step=500)
    stride = max(1, len(data) // max_points)
    plot_data = data.iloc[::stride]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_data["time"], plot_data[selected_channel], linewidth=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{selected_channel} — downsampled view")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    selected_gesture = st.selectbox(
        "Filter samples by gesture",
        options=sorted(data["class"].unique()),
        format_func=lambda x: f"{x} — {GESTURE_NAMES.get(int(x), 'Unknown')}"
    )
    gdata = data[data["class"] == selected_gesture].head(3000)
    if len(gdata):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(gdata["time"], gdata[selected_channel], linewidth=0.7)
        ax.set_title(f"{selected_channel}: {GESTURE_NAMES.get(int(selected_gesture))}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.subheader("Train a gesture classifier")
    st.write("Features: mean absolute value (MAV), RMS, variance, and waveform length for each channel.")
    if st.button("Train / evaluate Random Forest", type="primary"):
        with st.spinner("Extracting features and training model..."):
            train_data = data[data["class"].isin(range(1, 8))].reset_index(drop=True)
            X, y = extract_features(train_data, int(window_size))
        if len(X) < 10:
            st.error("Not enough same-label windows were found. Try a smaller window size or check the file.")
        elif len(np.unique(y)) < 2:
            st.error("At least two gesture classes are needed to train a classifier.")
        else:
            class_counts = pd.Series(y).value_counts()
            if class_counts.min() < 2:
                st.error("Some classes have fewer than two windows. Use more data or a smaller window size.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=15, random_state=42,
                    n_jobs=-1, class_weight="balanced"
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                score = accuracy_score(y_test, pred)
                st.session_state["emg_model"] = model
                st.session_state["feature_columns"] = X.columns.tolist()
                st.session_state["model_accuracy"] = score
                st.session_state["X"] = X
                st.session_state["y"] = y
                st.session_state["y_test"] = y_test
                st.session_state["pred"] = pred
                st.success("Model trained.")
    if "emg_model" in st.session_state:
        st.metric("Test accuracy", f"{st.session_state['model_accuracy'] * 100:.2f}%")
        y_test = st.session_state["y_test"]
        pred = st.session_state["pred"]
        present = sorted(set(y_test) | set(pred))
        st.text("Classification report")
        st.code(classification_report(
            y_test, pred, labels=present,
            target_names=[GESTURE_NAMES.get(int(c), str(c)) for c in present],
            zero_division=0
        ))
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, labels=present,
            display_labels=[GESTURE_NAMES.get(int(c), str(c)) for c in present],
            xticks_rotation=40, cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title("Confusion matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        import joblib
        model_bytes = None
        import io
        buffer = io.BytesIO()
        joblib.dump({
            "model": st.session_state["emg_model"],
            "feature_columns": st.session_state["feature_columns"],
            "gesture_names": GESTURE_NAMES,
            "window_size": int(window_size),
            "channels": CHANNELS
        }, buffer)
        st.download_button(
            "Download trained model",
            data=buffer.getvalue(),
            file_name="prosthetic_hand_emg_model.joblib",
            mime="application/octet-stream"
        )
    else:
        st.info("Click **Train / evaluate Random Forest** to run the analysis.")

with tab4:
    st.subheader("Gesture-to-action simulator")
    st.caption("Simulation only — no real servo is connected.")
    mapping = {
        1: ("Rest", "Relax fingers", "🖐️"),
        2: ("Fist", "Close fingers", "✊"),
        3: ("Wrist Flexion", "Tilt wrist forward", "↘️"),
        4: ("Wrist Extension", "Tilt wrist backward", "↗️"),
        5: ("Radial Deviation", "Tilt wrist toward thumb", "👉"),
        6: ("Ulnar Deviation", "Tilt wrist toward little finger", "👈"),
        7: ("Extended Palm", "Open fingers", "🖐️")
    }
    chosen = st.selectbox(
        "Choose a gesture to simulate",
        options=list(range(1, 8)),
        format_func=lambda x: f"{x} — {GESTURE_NAMES[x]}"
    )
    label, action, emoji = mapping[chosen]
    st.markdown(f"## {emoji} {label}")
    st.success(f"Simulated action: **{action}**")
    st.write("Suggested demo sequence: Rest → Fist → Extended Palm.")
    if st.button("Run demo sequence"):
        st.write("1. Rest — relax fingers")
        st.write("2. Fist — close fingers")
        st.write("3. Extended Palm — open fingers")
        st.caption("These are illustrative UI actions, not physical movements.")

st.divider()
st.caption("Research prototype. Random train/test splitting can overestimate performance because nearby EMG windows may be similar. Validate on separate recordings/subjects before making real-time control claims. Do not connect this dashboard directly to servos without safe power, movement limits, smoothing, and a fail-safe rest state.")
