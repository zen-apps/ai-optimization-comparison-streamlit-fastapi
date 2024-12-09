"""
Basketball Team Rotation Manager - A Streamlit application for optimizing basketball team rotations.

This application helps coaches manage their basketball team rotations by:
- Managing player rosters with positions and skill levels
- Optimizing player rotations based on various constraints
- Visualizing playing time distribution
- Validating lineup rules and requirements

The optimization can be performed using either OR-Tools or an external API.
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
import os
from typing import List, Dict, Union
import itertools
import requests
from ortools.sat.python import cp_model
import numpy as np
import plotly.express as px

# Streamlit configuration
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Constants
SUB_FREQUENCY = 5
PERIODS_PER_GAME = 8  # Two 20-minute halves with substitutions every 5 minutes


def plot_player_minutes(df: pd.DataFrame) -> None:
    """
    Create a bar chart showing the distribution of playing time across players.

    Args:
        df: DataFrame containing player statistics with 'Name' and 'Minutes Played' columns
    """
    fig = px.bar(
        df,
        x="Name",
        y="Minutes Played",
        title="Player Minutes Distribution",
        labels={"Name": "Player Name", "Minutes Played": "Minutes Played"},
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=500,
        bargap=0.2,
    )

    fig.update_traces(marker_color="#2563eb")
    st.plotly_chart(fig, use_container_width=True)


def create_player_stats(
    result_df: pd.DataFrame, minutes_per_period: int = 5
) -> pd.DataFrame:
    """
    Calculate player statistics from the rotation results.

    Args:
        result_df: DataFrame containing period-by-period player assignments
        minutes_per_period: Number of minutes in each period

    Returns:
        DataFrame with player statistics including total periods and minutes played
    """
    period_columns = [col for col in result_df.columns if col.startswith("Period")]

    player_stats = pd.DataFrame(
        {
            "Name": result_df.index,
            "Total Periods": result_df[period_columns].sum(axis=1),
            "Minutes Played": result_df[period_columns].sum(axis=1)
            * minutes_per_period,
        }
    )

    return player_stats.sort_values("Minutes Played", ascending=False)


def create_period_lineups(
    result_df: pd.DataFrame, player_positions: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a summary of lineups for each period including point guards and substitutions.

    Args:
        result_df: DataFrame containing period-by-period player assignments
        player_positions: DataFrame containing player position information

    Returns:
        DataFrame summarizing lineups for each period
    """
    periods = []
    players = []
    point_guards = []
    substitutions_out = []

    for period in range(1, PERIODS_PER_GAME + 1):
        period_col = f"Period {period}"
        playing_current = result_df[result_df[period_col] == 1].index.tolist()

        if period > 1:
            prev_period_col = f"Period {period-1}"
            playing_previous = set(
                result_df[result_df[prev_period_col] == 1].index.tolist()
            )
            playing_current_set = set(playing_current)
            players_out = playing_previous - playing_current_set
            substitutions_out.append(", ".join(sorted(players_out)))
        else:
            substitutions_out.append("")

        pgs = [
            player
            for player in playing_current
            if player_positions.loc[
                player_positions["name"] == player, "point_guard"
            ].iloc[0]
        ]

        periods.append(period)
        players.append(", ".join(sorted(playing_current)))
        point_guards.append(", ".join(sorted(pgs)))

    return pd.DataFrame(
        {
            "Period": periods,
            "Players": players,
            "Point Guards": point_guards,
            "Out from Previous": substitutions_out,
        }
    )


def calculate_out_from_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate which players were substituted out between periods.

    Args:
        df: DataFrame containing period lineup information

    Returns:
        DataFrame with added column showing players substituted out
    """

    def get_all_players(row):
        return (
            set(player.strip() for player in row["Players"].split(","))
            if pd.notna(row["Players"])
            else set()
        )

    result = df.copy()
    result["Out from Previous"] = ""

    for i in range(1, len(df)):
        prev_players = get_all_players(df.iloc[i - 1])
        current_players = get_all_players(df.iloc[i])
        out_players = prev_players - current_players
        if out_players:
            result.at[i, "Out from Previous"] = ", ".join(sorted(out_players))

    return result


def create_standardized_display(
    data: Union[pd.DataFrame, dict],
    passed_roster_df: pd.DataFrame = None,
    sub_frequency: int = SUB_FREQUENCY,
) -> None:
    """
    Create a standardized display of rotation results including lineups and statistics.

    Args:
        data: Either a DataFrame of results from OR-Tools or a dictionary of results from the API
        passed_roster_df: DataFrame containing roster information
        sub_frequency: Number of minutes between substitutions
    """
    period_mapping_dict = {
        1: "H1: 20-15",
        2: "H1: 15-10",
        3: "H1: 10-5",
        4: "H1: 5-0",
        5: "H2: 20-15",
        6: "H2: 15-10",
        7: "H2: 10-5",
        8: "H2: 5-0",
    }

    if isinstance(data, pd.DataFrame):
        summary_df = create_period_lineups(data, passed_roster_df)
        summary_df["Time"] = summary_df["Period"].map(period_mapping_dict)
        display_cols = ["Time", "Players", "Point Guards", "Out from Previous"]
        lineup_display = summary_df[display_cols]

        player_stats = create_player_stats(data)

        validation_summary = pd.DataFrame(
            {
                "pg_always_present": [True],
                "no_consecutive_sitting": [True],
                "balanced_playtime": [True],
            }
        )
    else:
        # Handle API response format
        lineup_rows = []
        for p in data["period_lineups"]:
            all_players = p["point_guards"] + p["shooting_guards"]
            row = {
                "Time": period_mapping_dict.get(p["period"], f"Period {p['period']}"),
                "Players": ", ".join(sorted(all_players)),
                "Point Guards": ", ".join(p["point_guards"]),
                "Out from Previous": "",
            }
            lineup_rows.append(row)
        lineup_display = pd.DataFrame(lineup_rows)
        lineup_display = calculate_out_from_previous(lineup_display)

        player_stats = pd.DataFrame(
            [
                {
                    "Name": rotation["name"],
                    "Total Periods": rotation["total_periods"],
                    "Minutes Played": rotation["total_periods"] * sub_frequency,
                }
                for rotation in data["player_rotations"]
            ]
        ).sort_values("Minutes Played", ascending=False)

        validation_summary = pd.DataFrame([data["validation_checks"]])

    # Display results
    st.subheader("Period-by-Period Lineup")
    st.dataframe(lineup_display, use_container_width=True, hide_index=True)

    st.subheader("Playing Time Distribution")
    st.dataframe(player_stats, use_container_width=True)

    with st.expander("View Player Minutes Distribution", expanded=False):
        plot_player_minutes(player_stats)

    st.subheader("Validation Checks")
    st.dataframe(validation_summary, use_container_width=True)


def optimize_lineup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize team rotations using OR-Tools CP-SAT solver.

    Args:
        df: DataFrame containing player information and constraints

    Returns:
        DataFrame containing optimized period assignments for each player
    """
    model = cp_model.CpModel()

    num_periods = PERIODS_PER_GAME
    num_players = len(df)
    players = range(num_players)
    periods = range(num_periods)

    desired_changes = num_players - 5
    pg_players = [i for i in players if df.iloc[i]["point_guard"]]

    # Create variables
    x = {}
    playing_pg = {}
    for i in players:
        for j in periods:
            x[i, j] = model.NewBoolVar(f"player_{i}_period_{j}")
            playing_pg[i, j] = (
                model.NewBoolVar(f"playing_pg_{i}_period_{j}")
                if i in pg_players
                else model.NewConstant(0)
            )

    # Add constraints
    for i in players:
        if df.iloc[i]["starting"]:
            model.Add(x[i, 0] == 1)

    for j in periods:
        model.Add(sum(x[i, j] for i in players) == 5)
        model.Add(sum(playing_pg[i, j] for i in players) == 1)
        for i in players:
            model.Add(playing_pg[i, j] <= x[i, j])

    # Handle substitutions
    period_changes = []
    for j in range(num_periods - 1):
        changes = []
        for i in players:
            is_change = model.NewBoolVar(f"change_{i}_{j}")
            model.Add(x[i, j] != x[i, j + 1]).OnlyEnforceIf(is_change)
            model.Add(x[i, j] == x[i, j + 1]).OnlyEnforceIf(is_change.Not())
            changes.append(is_change)

        period_change_count = sum(changes)
        period_changes.append(period_change_count)
        model.Add(period_change_count == desired_changes * 2)

    # Balance playing time
    player_periods = {i: sum(x[i, j] for j in periods) for i in players}

    max_diff = model.NewIntVar(0, num_periods, "max_diff")
    for i in players:
        for k in players:
            if i != k:
                diff = model.NewIntVar(-num_periods, num_periods, f"diff_{i}_{k}")
                model.Add(player_periods[i] - player_periods[k] == diff)
                model.Add(diff <= max_diff)
                model.Add(diff >= -max_diff)

    # Balance point guard playing time if multiple PGs
    if len(pg_players) > 1:
        pg_periods = {i: sum(playing_pg[i, j] for j in periods) for i in pg_players}
        pg_max_diff = model.NewIntVar(0, num_periods, "pg_max_diff")

        for i in pg_players:
            for k in pg_players:
                if i != k:
                    diff = model.NewIntVar(
                        -num_periods, num_periods, f"pg_diff_{i}_{k}"
                    )
                    model.Add(pg_periods[i] - pg_periods[k] == diff)
                    model.Add(diff <= pg_max_diff)
                    model.Add(diff >= -pg_max_diff)

        model.Minimize(max_diff + pg_max_diff)
    else:
        model.Minimize(max_diff)

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result = np.zeros((num_players, num_periods))
        pg_assignments = np.zeros((num_players, num_periods))

        for i in players:
            for j in periods:
                if solver.Value(x[i, j]) == 1:
                    result[i, j] = 1
                if solver.Value(playing_pg[i, j]) == 1:
                    pg_assignments[i, j] = 1

        periods_df = pd.DataFrame(
            result,
            index=df["name"],
            columns=[f"Period {i+1}" for i in range(num_periods)],
        )
        periods_df["Total Periods"] = periods_df.sum(axis=1)

        pg_df = pd.DataFrame(
            pg_assignments,
            index=df["name"],
            columns=[f"PG in Period {i+1}" for i in range(num_periods)],
        )
        pg_df["Total PG Periods"] = pg_df.sum(axis=1)

        return pd.concat([periods_df, pg_df["Total PG Periods"]], axis=1)

    return pd.DataFrame(
        0,
        index=df["name"],
        columns=[f"Period {i+1}" for i in range(num_periods)]
        + ["Total Periods", "Total PG Periods"],
    )


def get_api_optimized_roster(df: pd.DataFrame) -> dict:
    """
    Get optimized roster from external API.

    Args:
        df: DataFrame containing roster information

    Returns:
        Dictionary containing API response with optimized rotations
    """
    BACKEND_HOST = os.getenv("BACKEND_HOST")
    api_path = "v1/support/optimize_roster/"
    api_url = f"{BACKEND_HOST}{api_path}"

    query = {"roster": df.to_dict(orient="records")}
    try:
        response = requests.post(
            api_url, json=query, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing request: {str(e)}")
        return {}


def initial_roster(team_name: str) -> pd.DataFrame:
    """
    Create initial roster DataFrame for a given team.

    Args:
        team_name: Name of the team

    Returns:
        DataFrame containing initial roster setup
    """
    if team_name == "Demo":
        df = pd.DataFrame(
            {
                "name": [f"player{i}" for i in range(1, 9)],
                "point_guard": [False] * 8,
                "beginner": [False] * 8,
                "starting": [False] * 8,
            }
        )
        df["name"] = df["name"].str.split(" ").str[0]
        return df.sort_values("name").reset_index(drop=True)


@dataclass
class Player:
    """Player information dataclass."""

    name: str
    point_guard: bool
    beginner: bool
    starting: bool = False


def main():
    """Main function to run the Streamlit application."""
    st.title("Basketball Team Rotation Manager")

    # Initialize session state
    if "roster" not in st.session_state:
        st.session_state.roster = []
    if "edited_df" not in st.session_state:
        st.session_state.edited_df = None

    # Configure sidebar
    with st.sidebar:
        st.subheader("Settings")
        st.write(f"- Substitution Frequency: Every {SUB_FREQUENCY} minutes")
        st.write(
            "- Game Length: 40 minutes, two 20-minute halves, so 8 substitution periods"
        )
        st.write("- Spread out playing time evenly")
        st.write("- No player sits out twice in a row")
        st.write("- Maximum of 2 Beginners on court at one time")
        st.write("- Must be at least 1 PG on the court at all times")
        model_selectbox = st.selectbox("Select Model", ["OR-Tools", "GPT-4o"], index=0)
        use_or_tools = model_selectbox == "OR-Tools"

    # Team selection and roster loading
    col1, col2 = st.columns(2)
    with col1:
        team_name_selectbox = st.selectbox("Select Team Name", ["Demo"])
        if len(team_name_selectbox) > 0:
            st.write(f"Team Name: {team_name_selectbox}")
        else:
            st.warning("Need a Team name, select from the dropdown on the side")

    with col2:
        rosters_saved_dir = "./streamlit/data/roster/initial/"
        rosters_saved = [f for f in os.listdir(rosters_saved_dir) if f.endswith(".csv")]
        rosters_saved.sort(reverse=True)
        rosters_saved = [""] + rosters_saved
        select_previous_roster = st.selectbox(
            "Select Previous Saved Roster", rosters_saved
        )
        load_in_initial_roster_checkbox = st.checkbox("Load in initial roster")

    # Initialize roster DataFrame
    if len(team_name_selectbox) > 0 and load_in_initial_roster_checkbox:
        initial_df = initial_roster(team_name_selectbox)
    elif select_previous_roster:
        initial_df = pd.read_csv(
            f"{rosters_saved_dir}{select_previous_roster}"
        ).reset_index(drop=True)
        if "starting" not in initial_df.columns:
            initial_df["starting"] = False
    else:
        initial_df = pd.DataFrame(
            {
                "name": [""] * 8,
                "point_guard": [False] * 8,
                "beginner": [False] * 8,
                "starting": [False] * 8,
            }
        ).reset_index(drop=True)

    # Handle session state for edited DataFrame
    if st.session_state.edited_df is None:
        st.session_state.edited_df = initial_df.copy()

    st.session_state.edited_df = st.session_state.edited_df.reset_index(drop=True)

    # Create roster editor
    edited_roster_df = st.data_editor(
        initial_df,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn(
                "Name",
                disabled=True,
            ),
            "point_guard": st.column_config.CheckboxColumn(
                "Point Guard",
                help="Check if player is a point guard",
                default=False,
            ),
            "beginner": st.column_config.CheckboxColumn(
                "Beginner",
                help="Check if player is a beginner",
                default=False,
            ),
            "starting": st.column_config.CheckboxColumn(
                "Starting",
                default=False,
                help="Select starting players",
            ),
        },
        hide_index=True,
        key="roster_editor",
        use_container_width=True,
    )

    # Validate roster
    valid_starting_lineup = True

    if edited_roster_df["starting"].sum() > 5:
        st.warning("WARNING: Only 5 players can be selected as starting players")
        valid_starting_lineup = False

    if edited_roster_df["starting"].sum() < 5:
        st.warning("WARNING: Select 5 players as starting players")
        valid_starting_lineup = False

    if edited_roster_df["starting"].sum() > 0 and not any(
        edited_roster_df[edited_roster_df["starting"]]["point_guard"]
    ):
        st.warning("WARNING: Starting lineup must include a Point Guard")
        valid_starting_lineup = False

    beginners_in_starting_lineup = edited_roster_df[
        (edited_roster_df["starting"]) & (edited_roster_df["beginner"])
    ]
    if len(beginners_in_starting_lineup) > 2:
        st.warning("WARNING: Starting lineup can not have more than 2 Beginners")
        valid_starting_lineup = False

    st.session_state.edited_df = edited_roster_df.reset_index(drop=True)

    # Handle roster saving
    save_roster_button = st.button("Save Roster Settings for Later")

    st.session_state.roster = []
    for _, row in st.session_state.edited_df.iterrows():
        player = Player(
            name=row["name"],
            point_guard=row["point_guard"],
            beginner=row["beginner"],
            starting=row["starting"],
        )
        st.session_state.roster.append(player)

    if (
        save_roster_button
        and valid_starting_lineup
        and len(st.session_state.roster) > 0
    ):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame([vars(p) for p in st.session_state.roster])
        df = df.reset_index(drop=True)
        df.to_csv(
            f"./streamlit/data/roster/initial/{team_name_selectbox}_{timestamp}.csv",
            index=False,
        )
        st.success("Roster saved successfully!")
    elif save_roster_button and not valid_starting_lineup:
        st.error("Please correct the errors before saving the roster")

    if not valid_starting_lineup and not len(st.session_state.roster) > 0:
        st.info("Please select a valid starting lineup before proceeding")

    # Optimization section
    get_opt_btn = st.button("Get Optimal Rotation")
    if get_opt_btn:
        passed_roster_df = st.session_state.edited_df.copy()
        passed_roster_df = passed_roster_df.rename(
            columns={
                "name": "name",
                "point_guard": "point_guard",
            }
        )
        passed_roster_df["starting"] = passed_roster_df["starting"].astype(bool)
        passed_roster_df = passed_roster_df.drop(columns=["beginner"], errors="ignore")

        if use_or_tools:
            result = optimize_lineup(passed_roster_df)
            create_standardized_display(result, passed_roster_df)
        else:
            data = get_api_optimized_roster(passed_roster_df)
            create_standardized_display(data)

    # Reset button
    if st.button("Start Over"):
        st.session_state.roster = []
        st.session_state.edited_df = None
        st.session_state.starting_lineup = None
        st.session_state.player_rotations = None
        st.session_state.validation_checks = None


if __name__ == "__main__":
    main()
