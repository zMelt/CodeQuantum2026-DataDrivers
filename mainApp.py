import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DataDrivers – F1 Prediction", page_icon="🏎️", layout="wide")
st.title("DataDrivers")
st.header("F1 Win Probability Simulator")


# ── Data loading ───────────────────────────────────────────────────────────────
BASE_URL = "https://raw.githubusercontent.com/SZRoberson/CQ2026_DataTrack/main/data/"

@st.cache_data(show_spinner="Loading F1 data...")
def load_data():
    lap_df       = pd.read_csv(BASE_URL + "LapTimes.csv")
    race_results = pd.read_csv(BASE_URL + "RaceResults.csv")
    lap_df["LapTime"] = pd.to_timedelta(lap_df["LapTime"])

    # ── Feature engineering (mirrors vroomies.ipynb) ──────────────────────────
    def to_min(t):
        try:
            return pd.to_timedelta(t).total_seconds() / 60
        except:
            return None

    def get_fastest_qualifying(row):
        times = [to_min(row[col]) for col in ["Q1", "Q2", "Q3"]]
        times = [t for t in times if t is not None]
        return min(times) if times else None

    race_results["Qual"]   = race_results.apply(get_fastest_qualifying, axis=1)
    race_results["Race"]   = race_results["ElapsedTime"].apply(to_min)
    race_results["Won"]    = (race_results["Position"] == 1).astype(int)
    race_results["Podium"] = (race_results["Position"] <= 3).astype(int)
    race_results["Top5"]   = (race_results["Position"] <= 5).astype(int)
    race_results["Points"] = race_results["Points"].fillna(0)

    return lap_df, race_results

lap_df, race_results = load_data()

# ── vroomies.ipynb functions (ported verbatim) ─────────────────────────────────

def prepare_features_all(df):
    """Prepare features for all historical data."""
    features = pd.DataFrame()
    features["qualifying_time"] = df["Qual"]
    features["grid_position"]   = df["GridPosition"]
    features["race_round"]      = df["Round"]
    features = features.fillna(features.mean())
    return features


@st.cache_data(show_spinner="Training models…")
def train_historical_models(_df_historical, n_bootstrap=200):
    """Train bootstrapped logistic regression models on historical race data."""
    X_historical = prepare_features_all(_df_historical)
    models = {}
    targets = {
        "Win (1st)":       "Won",
        "Podium (1st-3rd)": "Podium",
        "Top 5":           "Top5",
    }
    for target_name, target_col in targets.items():
        y_historical = _df_historical[target_col]
        if y_historical.sum() < 3 or (len(y_historical) - y_historical.sum()) < 3:
            models[target_name] = None
            continue

        model_list, scaler_list = [], []
        for i in range(n_bootstrap):
            indices = resample(range(len(X_historical)), n_samples=len(X_historical),
                               replace=True, random_state=i)
            X_boot = X_historical.iloc[indices]
            y_boot = y_historical.iloc[indices]
            if len(np.unique(y_boot)) < 2:
                continue
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X_boot)
            model   = LogisticRegression(max_iter=1000, random_state=i)
            model.fit(X_scaled, y_boot)
            model_list.append(model)
            scaler_list.append(scaler)

        models[target_name] = {
            "models":   model_list,
            "scalers":  scaler_list,
            "features": X_historical.columns.tolist(),
        }
    return models


def predict_race_outcomes(race_data, historical_models):
    """Predict Win / Podium / Top-5 probabilities for each driver in a race."""
    X_race = prepare_features_all(race_data)
    all_predictions = {}

    for target_name, model_dict in historical_models.items():
        if model_dict is None:
            # Heuristic fallback
            rc = race_data.copy()
            rc["prob"] = 1 / (rc["GridPosition"] + 3)
            rc["prob"] = rc["prob"] / rc["prob"].max()
            pred_df = rc[["FullName", "DriverId", "TeamName", "GridPosition", "prob"]].copy()
            pred_df["probability"] = pred_df["prob"]
            pred_df["prob_lower"]  = pred_df["prob"]
            pred_df["prob_upper"]  = pred_df["prob"]
            all_predictions[target_name] = pred_df.sort_values("probability", ascending=False)
            continue

        all_probs = []
        for model, scaler in zip(model_dict["models"], model_dict["scalers"]):
            X_scaled = scaler.transform(X_race)
            all_probs.append(model.predict_proba(X_scaled)[:, 1])

        if not all_probs:
            continue

        all_probs  = np.array(all_probs)
        mean_probs = np.mean(all_probs, axis=0)
        lower_ci   = np.percentile(all_probs, 2.5,  axis=0)
        upper_ci   = np.percentile(all_probs, 97.5, axis=0)

        pred_df = race_data[["FullName", "DriverId", "TeamName", "GridPosition"]].copy()
        pred_df["probability"] = mean_probs
        pred_df["prob_lower"]  = lower_ci
        pred_df["prob_upper"]  = upper_ci
        all_predictions[target_name] = pred_df.sort_values("probability", ascending=False)

    return all_predictions


def filter_predictions(predictions, filter_type, filter_value):
    """Filter predictions by Driver or Team (mirrors vroomies.ipynb)."""
    filtered = {}
    for target_name, pred_df in predictions.items():
        if filter_type == "Driver":
            filtered[target_name] = pred_df[pred_df["FullName"] == filter_value].copy()
        elif filter_type == "Team":
            filtered[target_name] = pred_df[pred_df["TeamName"] == filter_value].copy()
        else:
            filtered[target_name] = pred_df.copy()
    return filtered


def create_figure_list(predictions, race_data, filter_type="All", filter_value=None):
    """
    Build the figure_list exactly as in vroomies.ipynb.

    filter_type : "All" | "Driver" | "Team"
    Returns a list of up to 4–6 Plotly figures depending on filter mode.
    """
    figure_list = []

    if filter_type != "All" and filter_value:
        predictions = filter_predictions(predictions, filter_type, filter_value)

    event_name = race_data["Event Name"].iloc[0]

    # ── Figure 1: Win Probability ──────────────────────────────────────────────
    if "Win (1st)" in predictions and len(predictions["Win (1st)"]) > 0:
        win_probs = predictions["Win (1st)"].head(15)
        if filter_type in ("Driver", "Team") and filter_value:
            title1 = f"Win Probability – {filter_value}"
        else:
            title1 = f"Win Probability – {event_name}"

        fig1 = px.bar(
            win_probs, x="FullName", y="probability",
            title=title1,
            labels={"probability": "Probability", "FullName": "Driver"},
            color="probability", color_continuous_scale="Viridis",
            text=win_probs["probability"].apply(lambda x: f"{x:.1%}"),
        )
        fig1.update_layout(height=500, showlegend=False)
        fig1.update_traces(textposition="outside")
        figure_list.append(fig1)

    # ── Figure 2: Podium Probability ───────────────────────────────────────────
    if "Podium (1st-3rd)" in predictions and len(predictions["Podium (1st-3rd)"]) > 0:
        podium_probs = predictions["Podium (1st-3rd)"].head(15)
        if filter_type in ("Driver", "Team") and filter_value:
            title2 = f"Podium Probability – {filter_value}"
        else:
            title2 = f"Podium Probability – {event_name}"

        fig2 = px.bar(
            podium_probs, x="FullName", y="probability",
            title=title2,
            labels={"probability": "Probability", "FullName": "Driver"},
            color="probability", color_continuous_scale="Plasma",
            text=podium_probs["probability"].apply(lambda x: f"{x:.1%}"),
        )
        fig2.update_layout(height=500, showlegend=False)
        fig2.update_traces(textposition="outside")
        figure_list.append(fig2)

    # ── Figure 3: Top 5 Probability ────────────────────────────────────────────
    if "Top 5" in predictions and len(predictions["Top 5"]) > 0:
        top5_probs = predictions["Top 5"].head(15)
        if filter_type in ("Driver", "Team") and filter_value:
            title3 = f"Top 5 Probability – {filter_value}"
        else:
            title3 = f"Top 5 Probability – {event_name}"

        fig3 = px.bar(
            top5_probs, x="FullName", y="probability",
            title=title3,
            labels={"probability": "Probability", "FullName": "Driver"},
            color="probability", color_continuous_scale="Inferno",
            text=top5_probs["probability"].apply(lambda x: f"{x:.1%}"),
        )
        fig3.update_layout(height=500, showlegend=False)
        fig3.update_traces(textposition="outside")
        figure_list.append(fig3)

    # ── Figure 4a: Win Probabilities with 95% CI (All view) ───────────────────
    if filter_type == "All" and "Win (1st)" in predictions:
        top_drivers = predictions["Win (1st)"].head(8).copy()

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=top_drivers["FullName"],
            y=top_drivers["probability"],
            name="Probability",
            marker_color="steelblue",
            text=top_drivers["probability"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
            error_y=dict(
                type="data", symmetric=False,
                array=top_drivers["prob_upper"] - top_drivers["probability"],
                arrayminus=top_drivers["probability"] - top_drivers["prob_lower"],
                color="gray", thickness=1,
            ),
        ))
        fig4.update_layout(
            title=f"Win Probabilities with 95% Confidence Intervals – {event_name}",
            xaxis_title="", yaxis_title="Probability",
            height=500, showlegend=False,
        )
        figure_list.append(fig4)

    # ── Figure 4b: Outcome Probabilities for a single Driver ──────────────────
    if filter_type == "Driver" and filter_value and "Win (1st)" in predictions:
        driver_data = predictions["Win (1st)"]
        if len(driver_data) > 0:
            driver_row = driver_data.iloc[0]
            categories = ["Win", "Podium", "Top 5"]
            values = [
                driver_row["probability"] * 100,
                predictions["Podium (1st-3rd)"].iloc[0]["probability"] * 100
                    if "Podium (1st-3rd)" in predictions and len(predictions["Podium (1st-3rd)"]) > 0 else 0,
                predictions["Top 5"].iloc[0]["probability"] * 100
                    if "Top 5" in predictions and len(predictions["Top 5"]) > 0 else 0,
            ]
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=categories, y=values,
                marker_color=["#FEC11A", "#C0C0C0", "#CD7F32"],
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
            ))
            fig5.update_layout(
                title=f"{filter_value} – Outcome Probabilities",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                height=400,
            )
            figure_list.append(fig5)

    # ── Figure 4c: Team driver comparison ─────────────────────────────────────
    if filter_type == "Team" and filter_value and "Win (1st)" in predictions:
        team_data = predictions["Win (1st)"]
        if len(team_data) > 0:
            team_drivers = team_data[["FullName", "probability"]].copy()
            if "Podium (1st-3rd)" in predictions:
                team_drivers = team_drivers.copy()
                team_drivers["podium"] = predictions["Podium (1st-3rd)"]["probability"].values
            if "Top 5" in predictions:
                team_drivers["top5"] = predictions["Top 5"]["probability"].values

            fig6 = go.Figure()
            fig6.add_trace(go.Bar(name="Win",    x=team_drivers["FullName"],
                                  y=team_drivers["probability"] * 100, marker_color="#FFD700"))
            if "podium" in team_drivers.columns:
                fig6.add_trace(go.Bar(name="Podium", x=team_drivers["FullName"],
                                      y=team_drivers["podium"] * 100, marker_color="#C0C0C0"))
            if "top5" in team_drivers.columns:
                fig6.add_trace(go.Bar(name="Top 5",  x=team_drivers["FullName"],
                                      y=team_drivers["top5"] * 100, marker_color="#CD7F32"))
            fig6.update_layout(
                title=f"{filter_value} – Racer Comparison",
                yaxis_title="Probability (%)",
                barmode="group", height=500,
            )
            figure_list.append(fig6)

    return figure_list


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Simulation Settings")

all_events = sorted(race_results["Event Name"].dropna().unique().tolist())
selected_event = st.sidebar.selectbox(
    "Track",
    options=all_events,
    index=len(all_events) - 1,
    help="The race used for predictions. Training uses all earlier rounds.",
)

n_bootstrap = st.sidebar.select_slider(
    "Bootstrap Iterations",
    # values from 25 to 500 in increments of 5
    options=list(range(25, 505, 5)),
    value=250,
    help="More iterations = more stable probabilities, but slower.",
)

run_btn = st.sidebar.button("▶ Run Prediction", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("Results Filter")

filter_mode = st.sidebar.radio(
    "Show",
    options=["No Filter", "All Racers", "By Racer", "By Team"],
    help=(
        "No Filter / All Racer → ranked bar charts for everyone.\n"
        "By Racer → spotlights one racer across all 4 outcome types.\n"
        "By Team → compares both team drivers side-by-side."
    ),
)

filter_value = None
if filter_mode == "By Racer":
    all_racers  = sorted(race_results["FullName"].dropna().unique().tolist())
    filter_value = st.sidebar.selectbox("Select Racer", options=all_racers)
elif filter_mode == "By Team":
    all_teams    = sorted(race_results["TeamName"].dropna().unique().tolist())
    filter_value = st.sidebar.selectbox("Select Team", options=all_teams)

# map sidebar filter_mode → vroomies filter_type string
FILTER_TYPE_MAP = {
    "No Filter":   "All",
    "All Racers": "All",
    "By Racer":   "Driver",
    "By Team":     "Team",
}
filter_type = FILTER_TYPE_MAP[filter_mode]

# ── Status caption ─────────────────────────────────────────────────────────────
st.caption(
    f"Event: **{selected_event}** · "
    f"**{n_bootstrap}** bootstrap iterations · "
    f"Filter: **{filter_mode}**"
    + (f" → *{filter_value}*" if filter_value else "")
)

# ── Run pipeline ───────────────────────────────────────────────────────────────
cache_key = (selected_event, n_bootstrap)

if "predictions" not in st.session_state or \
   st.session_state.get("cache_key") != cache_key or \
   run_btn:

    # Determine the round number for the selected event
    target_round = race_results.loc[
        race_results["Event Name"] == selected_event, "Round"
    ].iloc[0]

    df_clean = race_results.dropna(subset=["Qual", "Race"]).copy()

    historical_data  = df_clean[df_clean["Round"] < target_round].copy()
    race_to_predict  = df_clean[df_clean["Round"] == target_round].copy()

    if historical_data.empty:
        st.error("Not enough historical races before this event to train a model. "
                 "Please select a later Track.")
        st.stop()

    if race_to_predict.empty:
        st.error("No qualifying/race data found for this event.")
        st.stop()

    with st.spinner("Training models and generating predictions…"):
        historical_models = train_historical_models(historical_data, n_bootstrap=n_bootstrap)
        predictions       = predict_race_outcomes(race_to_predict, historical_models)

    st.session_state.predictions      = predictions
    st.session_state.race_to_predict  = race_to_predict
    st.session_state.cache_key        = cache_key

predictions     = st.session_state.predictions
race_to_predict = st.session_state.race_to_predict

# ── Build figure_list (mirrors vroomies.ipynb exactly) ────────────────────────
figure_list = create_figure_list(
    predictions, race_to_predict,
    filter_type=filter_type,
    filter_value=filter_value,
)

# ── Display all figures ────────────────────────────────────────────────────────
if not figure_list:
    st.warning(
        f"No figures to display for **{filter_value}** with the current filter. "
        "They may not have qualifying data for this event. Try 'No Filter' or a different event."
    )
else:
    TITLES = {
        "All": [
            "Win Probability",
            "Podium Probability (Top 3)",
            "Podium Probability (Top 5)",
            "Win Probabilities (95% Confidence Interval)",
        ],
        "Racer": [
            "Win Probability",
            "Podium Probability (Top 3)",
            "Podium Probability (Top 5)",
            "Outcome Probabilities",
        ],
        "Team": [
            "Win Probability",
            "Podium Probability (Top 3)",
            "Podium Probability (Top 5)",
            "Racer Comparison",
        ],
    }
    section_titles = TITLES.get(filter_type, TITLES["All"])

    for i, fig in enumerate(figure_list):
        label = section_titles[i] if i < len(section_titles) else f"Figure {i + 1}"
        st.subheader(label)
        st.plotly_chart(fig, use_container_width=True)

# ── Raw predictions table ──────────────────────────────────────────────────────
with st.expander("Full prediction table (Win probability, all drivers)"):
    if "Win (1st)" in predictions:
        tbl = predictions["Win (1st)"][["FullName", "TeamName", "GridPosition", "probability",
                                        "prob_lower", "prob_upper"]].copy()
        tbl.columns = ["Racer", "Team", "Grid", "Win %", "Lower 95%", "Upper 95%"]
        st.dataframe(
            tbl.style.format({
                "Win Prob": "{:.2%}", "Lower 95%": "{:.2%}", "Upper 95%": "{:.2%}"
            }),
            use_container_width=True, hide_index=True,
        )

# ── Search lookup ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Search for Racer")

if "Win (1st)" in predictions:
    all_names     = predictions["Win (1st)"]["FullName"].tolist()
    selected_name = st.selectbox("Select entry", options=all_names)

    if selected_name:
        row = predictions["Win (1st)"][predictions["Win (1st)"]["FullName"] == selected_name].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Racer",           row["FullName"])
        c2.metric("Team",             row["TeamName"])
        c3.metric("Grid Position",    str(int(row["GridPosition"])))
        c4.metric("Win Probability",  f"{row['probability']:.2%}")
