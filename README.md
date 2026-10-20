# NeuroGrip — EMG Prosthetic Hand Lab

A Streamlit dashboard for exploring recorded 8-channel MYO EMG data, extracting simple time-domain features, training a Random Forest baseline, and previewing gesture-to-action mappings.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run emg_prosthetic_dashboard.py
```

## Deploy

Upload the contents of this folder to the root of your GitHub repository. Rename `emg_prosthetic_dashboard_v2.py` to `emg_prosthetic_dashboard.py` (or change the Streamlit Cloud main-file path to match), and rename `requirements_v2.txt` to `requirements.txt`. Then redeploy/reboot your Streamlit Community Cloud app.

The dashboard uses uploaded recordings. It does not stream from the MYO bracelet or control physical servos. Model scores from random splits can be optimistic; validate on separate sessions/subjects.
