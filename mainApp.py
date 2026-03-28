import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict
from sklearn.utils import resample
from vroomies import remove_outliers, prepare_simulation_data, run_bootstrapped_simulations, calculate_win_probabilities, plot_win_probabilities


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DataDrivers – F1 Prediction", page_icon="🏎️", layout="wide")
st.title("DataDrivers")
st.header("F1 Win Probability Simulator")

# ── Data loading (cached so it only runs once) ─────────────────────────────────
BASE_URL = "https://raw.githubusercontent.com/SZRoberson/CQ2026_DataTrack/main/data/"

@st.cache_data(show_spinner="Loading F1 data...")
def load_data():
    lap_df  = pd.read_csv(BASE_URL + "LapTimes.csv")
    race_df = pd.read_csv(BASE_URL + "RaceResults.csv")
    lap_df["LapTime"] = pd.to_timedelta(lap_df["LapTime"])
    return lap_df, race_df

lap_df, race_df = load_data()

# ── Functions from vroomies.ipynb ──────────────────────────────────────────────

'''
def remove_outliers(laptimes_observations):
    """Remove lap time outliers using the IQR method."""
    cleaned = {}
    for entity_id, laptimes in laptimes_observations.items():
        if len(laptimes) < 4:
            cleaned[entity_id] = laptimes
            continue
        arr = np.array(laptimes)
        Q1, Q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        IQR = Q3 - Q1
        mask = (arr >= Q1 - 1.5 * IQR) & (arr <= Q3 + 1.5 * IQR)
        cleaned[entity_id] = arr[mask].tolist()
    return cleaned
'''

def prepare_simulation_data(lap_df, race_df, selected_event=None,
                             analysis_type="driver", apply_outlier_removal=False):
    """Load, filter, and structure lap times for drivers or teams."""
    processed_lap  = lap_df.copy()
    processed_race = race_df.copy()

    if selected_event:
        event_races  = processed_race[processed_race["Event Name"] == selected_event]
        participants = event_races["DriverNumber"].unique()
        processed_lap  = processed_lap[processed_lap["DriverNumber"].isin(participants)]
        processed_race = event_races

    id_col, name_col = ("DriverNumber", "FullName") if analysis_type == "driver" else ("TeamId", "TeamName")

    name_map = (
        processed_race[[id_col, name_col]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)[name_col]
        .to_dict()
    )

    # lap_df only has DriverNumber — for team analysis, join TeamId in from race_df
    if analysis_type == "team":
        driver_to_team = (
            processed_race[["DriverNumber", "TeamId", "TeamName"]]
            .drop_duplicates(subset=["DriverNumber"])
        )
        processed_lap = processed_lap.merge(driver_to_team, on="DriverNumber", how="left")

    laptimes_obs = {}
    for entity_id in processed_lap[id_col].unique():
        laps_s = processed_lap[processed_lap[id_col] == entity_id]["LapTime"].dt.total_seconds().dropna()
        if len(laps_s) > 0:
            laptimes_obs[entity_id] = laps_s.tolist()

    if apply_outlier_removal:
        laptimes_obs = remove_outliers(laptimes_obs)

    return laptimes_obs, name_map


def simulate_bootstrapped_race(driver_laptimes, num_laps=60):
    """Simulate one race by bootstrap-sampling lap times."""
    if not driver_laptimes:
        return float("inf")
    return sum(resample(driver_laptimes, n_samples=num_laps, replace=True))


def run_bootstrapped_simulations(laptimes_obs_dict, num_laps, num_simulations):
    """Run the full bootstrap simulation loop and return win counts."""
    win_counts = defaultdict(int)
    for _ in range(num_simulations):
        results = {eid: simulate_bootstrapped_race(lt, num_laps) for eid, lt in laptimes_obs_dict.items()}
        if results:
            win_counts[min(results, key=results.get)] += 1
    return win_counts


def calculate_win_probabilities(win_counts, num_simulations, name_map, target_id=None):
    """Convert win counts to a sorted probability DataFrame, or a single float if target_id given."""
    probs = {eid: cnt / num_simulations for eid, cnt in win_counts.items()}
    df = pd.DataFrame(list(probs.items()), columns=["ID", "WinProbability"])
    df["Name"] = df["ID"].map(name_map).fillna(df["ID"].astype(str))
    df = df.sort_values("WinProbability", ascending=False).reset_index(drop=True)

    if target_id is not None:
        row = df[df["ID"] == target_id]
        return float(row["WinProbability"].iloc[0]) if not row.empty else 0.0
    return df


def plot_win_probabilities(prob_df, title):
    """Return a Plotly bar chart of win probabilities."""
    fig = px.bar(
        prob_df, x="Name", y="WinProbability",
        title=title,
        labels={"Name": "Driver / Team", "WinProbability": "Win Probability"},
        color="Name", text="WinProbability", template="ggplot2",
    )
    fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis=dict(tickformat=".0%"),
        showlegend=False,
        height=520,
    )
    return fig

# ── Sidebar controls (parameters sent into the simulation) ─────────────────────
st.sidebar.title("⚙️ Simulation Settings")

analysis_type = st.sidebar.radio(
    "Analysis Type",
    options=["driver", "team"],
    format_func=lambda x: "Racer" if x == "driver" else "Team",
    help="Simulate individual racer probabilities or team probabilities.",
)

all_events    = sorted(race_df["Event Name"].dropna().unique().tolist())
selected_event_raw = st.sidebar.selectbox(
    "Event Filter",
    options=["All Events"] + all_events,
    help="Filter lap data to one Grand Prix, or use the full season.",
)
selected_event = None if selected_event_raw == "All Events" else selected_event_raw

num_laps = st.sidebar.slider(
    "Race Laps", min_value=30, max_value=78, value=60, step=1,
)

num_simulations = st.sidebar.select_slider(
    "Bootstrap Simulations",
    options=[100, 500, 1_000, 5_000, 10_000],
    value=1_000,
    help="More simulations = more accurate, but slower.",
)

apply_outlier_removal = st.sidebar.toggle(
    "Remove Outlier Laps (IQR)",
    value=False,
    help="Removes laps outside Q1 ± 1.5×IQR before simulating.",
)

run_btn = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Run simulation ─────────────────────────────────────────────────────────────
st.caption(
    f"**{analysis_type.title()}** · Event: **{selected_event_raw}** · "
    f"**{num_laps}** laps · **{num_simulations:,}** simulations · "
    f"Outlier removal: **{'On' if apply_outlier_removal else 'Off'}**"
)

if "results_df" not in st.session_state or run_btn:
    with st.spinner("Running bootstrap simulation…"):
        laptimes_obs, name_map = prepare_simulation_data(
            lap_df, race_df,
            selected_event=selected_event,
            analysis_type=analysis_type,
            apply_outlier_removal=apply_outlier_removal,
        )
        win_counts = run_bootstrapped_simulations(laptimes_obs, num_laps, num_simulations)
        prob_df    = calculate_win_probabilities(win_counts, num_simulations, name_map)

    st.session_state.results_df   = prob_df
    st.session_state.results_meta = dict(
        analysis_type=analysis_type,
        selected_event=selected_event_raw,
    )

prob_df = st.session_state.results_df
meta    = st.session_state.results_meta

# ── Chart ──────────────────────────────────────────────────────────────────────
label     = "Drivers" if meta["analysis_type"] == "driver" else "Constructors"
fig_title = f"Bootstrapped Win Probabilities – {label} ({meta['selected_event']})"
st.plotly_chart(plot_win_probabilities(prob_df, fig_title), use_container_width=True)

# ── Raw data table ─────────────────────────────────────────────────────────────
with st.expander("📋 Full probability table"):
    st.dataframe(
        prob_df[["Name", "WinProbability"]]
        .rename(columns={"Name": "Driver / Team", "WinProbability": "Win Probability"})
        .style.format({"Win Probability": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

# ── Single-entity lookup ───────────────────────────────────────────────────────
st.divider()
st.subheader("Search for Racer / Team")
t
selected_name = st.selectbox("Select entry", options=prob_df["Name"].tolist())
if selected_name:
    row = prob_df[prob_df["Name"] == selected_name].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Name",            row["Name"])
    c2.metric("ID",              str(row["ID"]))
    c3.metric("Win Probability", f"{row['WinProbability']:.2%}")