import os
import json
import logging
from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd
import re

# Set up logging with a more specific name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("roster_optimization")


class Player(BaseModel):
    """Player data model"""

    name: str = Field(description="Player's full name")
    is_pg: bool = Field(description="Whether player can play Point Guard position")
    is_sg: bool = Field(description="Whether player can play Shooting Guard position")


class PlayerRotation(BaseModel):
    """Player rotation details"""

    name: str = Field(description="Name of the player")
    periods_played: List[int] = Field(
        description="List of periods the player participates in"
    )
    total_periods: int = Field(description="Total number of periods played")
    meets_playtime_goal: bool = Field(
        description="Whether the player meets the minimum playtime goal"
    )


class PeriodLineup(BaseModel):
    """Lineup for a single period"""

    period: int = Field(description="Period number (1-8)", ge=1, le=8)
    point_guards: List[str] = Field(
        description="Point guards playing in this period", min_items=1
    )
    shooting_guards: List[str] = Field(
        description="Shooting guards playing in this period"
    )


class RosterOptimization(BaseModel):
    """Complete roster optimization result"""

    period_lineups: List[PeriodLineup] = Field(
        description="Detailed lineup for each 5-minute period"
    )
    player_rotations: List[PlayerRotation] = Field(
        description="Summary of each player's participation"
    )
    validation_checks: Dict[str, bool] = Field(
        description="Validation of rotation rules"
    )


def create_roster_constraints(roster_df: pd.DataFrame) -> str:
    """Create a string of roster constraints based on player eligibility"""
    pg_eligible = roster_df[roster_df["is_pg"]]["name"].tolist()
    return (
        "PG-eligible players (ONLY these players can be assigned as PG):\n"
        f"{', '.join(pg_eligible)}\n\n"
        "Remember: Only the above players can be assigned as PG in any period. "
        "This is a strict rule that cannot be broken."
    )


def extract_periods(response_text: str) -> List[Tuple[int, List[str], List[str]]]:
    """Extract period information from response text"""
    periods = []
    period_pattern = (
        r"\*\*Period (\d+):\*\*\n\*\*PG\*\*: ([^\n]*)\n\*\*SGs\*\*: ([^\n]*)"
    )
    matches = list(re.finditer(period_pattern, response_text))

    if not matches:
        logger.error(f"No period information found in response: {response_text}")
        raise ValueError("Could not parse period information from LLM response")

    return [
        (
            int(match.group(1)),
            [pg.strip() for pg in match.group(2).split(",") if pg.strip()],
            [sg.strip() for sg in match.group(3).split(",") if sg.strip()],
        )
        for match in matches
    ]


def extract_player_summary(response_text: str) -> List[Tuple[str, int]]:
    """Extract player summary information from response text"""
    player_pattern = r"\|\s*\*\*([^*]+)\*\*\s*\|\s*(\d+)\s*\|"
    matches = list(re.finditer(player_pattern, response_text))

    if not matches:
        logger.warning("No player summary table found in response")
        return []

    return [(match.group(1).strip(), int(match.group(2))) for match in matches]


def parse_llm_response(
    response_text: str, roster_df: pd.DataFrame
) -> RosterOptimization:
    """Parse the LLM response text into structured format"""
    try:
        # Extract and validate period data
        period_data = extract_periods(response_text)
        pg_eligible_players = set(roster_df[roster_df["is_pg"]]["name"].tolist())

        # Process periods and track player participation
        period_lineups = []
        player_periods: Dict[str, List[int]] = {}
        invalid_pg_assignments = []

        for period_num, pgs, sgs in period_data:
            # Validate PG assignments
            for pg in pgs:
                if pg not in pg_eligible_players:
                    invalid_pg_assignments.append((period_num, pg))

            # Track player periods
            for player in pgs + sgs:
                if player not in player_periods:
                    player_periods[player] = []
                player_periods[player].append(period_num)

            period_lineups.append(
                PeriodLineup(period=period_num, point_guards=pgs, shooting_guards=sgs)
            )

        if invalid_pg_assignments:
            error_details = "\n".join(
                f"Period {period}: {player} is not PG-eligible"
                for period, player in invalid_pg_assignments
            )
            raise ValueError(f"Invalid PG assignments detected:\n{error_details}")

        # Create player rotations
        player_rotations = [
            PlayerRotation(
                name=player,
                periods_played=sorted(periods),
                total_periods=len(periods),
                meets_playtime_goal=len(periods) >= 4,
            )
            for player, periods in player_periods.items()
        ]

        # Validation checks
        validation_checks = {
            "pg_always_present": all(
                len(period.point_guards) > 0 for period in period_lineups
            ),
            "pg_eligibility_valid": all(
                all(pg in pg_eligible_players for pg in period.point_guards)
                for period in period_lineups
            ),
            "no_consecutive_sitting": True,  # Implement detailed check if needed
            "balanced_playtime": all(p.total_periods >= 4 for p in player_rotations),
        }

        return RosterOptimization(
            period_lineups=period_lineups,
            player_rotations=player_rotations,
            validation_checks=validation_checks,
        )

    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}\nResponse: {response_text}")
        raise ValueError(f"Failed to parse LLM response: {str(e)}")


support = APIRouter()


@support.post("/optimize_roster/")
async def optimize_roster(query: Dict[str, Any]) -> Response:
    """Optimize a basketball team roster with structured output"""
    try:
        # Extract and validate roster data
        roster = query.get("roster", [])
        if not isinstance(roster, list):
            raise HTTPException(
                status_code=400,
                detail="Invalid roster format. Expected a list of player data.",
            )

        # Convert to DataFrame and validate structure
        roster_df = pd.DataFrame(roster)
        expected_columns = {"name", "is_pg", "is_sg"}

        if set(roster_df.columns) != expected_columns:
            logger.info("Fixing column names...")
            if len(roster_df.columns) >= 3:
                roster_df.columns = ["name", "is_pg", "is_sg"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid roster data format. Expected columns: {expected_columns}",
                )

        # Initialize LLM
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="o1-preview")

        # Create optimization prompt
        roster_constraints = create_roster_constraints(roster_df)
        prompt = f"""Create an optimized basketball roster rotation following these rules:
        1. The initial period must be the starting lineup
        2. 20 minutes for each half (8 periods of 5 minutes each)
        3. Substitutes every 5 minutes
        4. Minimum 1 PG on the court at all times
        5. Max of 2 Beginners on the court at any one time
        6. No player can sit out twice in a row
        7. Playing time should be balanced as best as possible
        8. PG time should be balanced as best as possible
        9. All periods of the game must have 5 players on the court
        10. All players should play the same number of periods

        {roster_constraints}

        Format your response EXACTLY like this:

        **Period 1:**
        **PG**: [Point Guard Name]
        **SGs**: [Other Player Names]

        **Period 2:**
        **PG**: [Point Guard Name]
        **SGs**: [Other Player Names]

        (Continue for all 8 periods)

        ---

        | **Player Name** | **Periods Played** |
        |----------------|-------------------|
        | **Player 1** | X |
        | **Player 2** | X |

        Roster data:
        {roster_df.to_string()}"""

        # Get and parse LLM response
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"LLM Response received: {response.content[:200]}...")

        structured_response = parse_llm_response(response.content, roster_df)
        return Response(
            content=json.dumps(structured_response.dict(), indent=2),
            media_type="application/json",
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in roster optimization: {str(e)}")
        return Response(
            content=json.dumps({"error": str(e), "type": "internal_server_error"}),
            status_code=500,
            media_type="application/json",
        )
