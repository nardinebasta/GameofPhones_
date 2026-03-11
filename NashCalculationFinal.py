"""
NashCalculationFinal.py

Computes Nash equilibrium strategies from empirical payoff matrices
using linear programming.

Implements the equilibrium computation described in Section 3.4
(Equations 6-7). For a zero-sum game with payoff matrix A:

    Scammer (maximiser):
        max_{x,v} v  s.t. v <= sum_i a_ij x_i  for all j,
                            sum_i x_i = 1, x_i >= 0

    Baiter (minimiser):
        min_{y,w} w  s.t. w >= sum_j a_ij y_j  for all i,
                            sum_j y_j = 1, y_j >= 0

At equilibrium, v* = w* defines the game value (expected per-turn
scammer payoff under optimal play by both sides).

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import os
import json
import numpy as np
from scipy.optimize import linprog


# Strategy names for readable output (Table 1)
PLAYER1_STRATEGIES = [
    "Urgency",               # S1
    "Authority",              # S2
    "Emotional Manipulation", # S3
    "Incentive",              # S4
    "Information Extraction", # S5
    "Technical Jargon",       # S6
    "Reassurance",            # S7
    "Building Rapport",       # S8
    "Persistence",            # S9
    "Hang Up",               # S10
]

PLAYER2_STRATEGIES = [
    "Delay",                  # B1
    "Obfuscation",           # B2
    "Technical Difficulty",   # B3
    "Counter Questioning",    # B4
    "Information Fabrication",# B5
    "Malicious Compliance",   # B6
    "Conversation Diversion", # B7
    "Pretended Naivety",      # B8
    "Verification Request",   # B9
    "Reverse Engineering",    # B10
]


def replace_strategy_names(file_path, output_file_path):
    """
    Replace numerical strategy identifiers with descriptive names
    in Nash equilibrium JSON output.

    Converts entries like "Strategy 1" to their descriptive names
    (e.g., "Urgency") for human-readable presentation of equilibrium
    distributions (Table 6 in the paper).

    Args:
        file_path (str): Path to the raw Nash equilibrium JSON file.
        output_file_path (str): Path to save the labelled output.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    # Replace numerical indices with strategy names
    data["Mixed strategy for scammer"] = {
        PLAYER1_STRATEGIES[int(k.split()[-1]) - 1]: v
        for k, v in data["Mixed strategy for scammer"].items()
    }
    data["Mixed strategy for victim"] = {
        PLAYER2_STRATEGIES[int(k.split()[-1]) - 1]: v
        for k, v in data["Mixed strategy for victim"].items()
    }

    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Updated results saved to {output_file_path}")


def save_game_results_to_json(game_value, player1_strategy,
                               player2_strategy, filename="game_results.json"):
    """
    Save Nash equilibrium results to a JSON file.

    Args:
        game_value (float): The game value v* = w* at equilibrium.
            Near zero indicates strategic parity; positive values
            indicate scammer advantage (Section 5.3).
        player1_strategy (np.array): Scammer's equilibrium mixed
            strategy (probability distribution over S1-S10).
        player2_strategy (np.array): Baiter's equilibrium mixed
            strategy (probability distribution over B1-B10).
        filename (str): Output filename for the JSON results.
    """
    game_results = {
        "Value of the game": game_value,
        "Mixed strategy for scammer": {
            f"Strategy {i + 1}": prob
            for i, prob in enumerate(player1_strategy)
        },
        "Mixed strategy for victim": {
            f"Strategy {i + 1}": prob
            for i, prob in enumerate(player2_strategy)
        },
    }

    with open(filename, "w") as file:
        json.dump(game_results, file, indent=4)

    print(f"Results have been saved to {filename}")


def nash(player1_file, player2_file, output_file):
    """
    Compute Nash equilibrium via linear programming (Equations 6-7).

    Loads the scammer's and baiter's payoff matrices from CSV,
    verifies the zero-sum property, and solves for the equilibrium
    mixed strategies using the HiGHS solver.

    The scammer's equilibrium is computed directly from the LP.
    The baiter's equilibrium is derived from the slack variables
    of the scammer's LP (dual relationship in zero-sum games).

    Args:
        player1_file (str): CSV file with the scammer's payoff matrix.
        player2_file (str): CSV file with the negated payoff matrix
            (baiter's perspective).
        output_file (str): Path to save the equilibrium results JSON.
    """
    # Load payoff matrices (skip header row and first column of labels)
    player1_matrix = np.genfromtxt(
        player1_file, delimiter=",", skip_header=1, usecols=range(1, 11)
    )
    player2_matrix = np.genfromtxt(
        player2_file, delimiter=",", skip_header=1, usecols=range(1, 11)
    )

    # Verify zero-sum property: player2's matrix should be the negation
    assert np.allclose(
        player1_matrix, -player2_matrix
    ), "The matrices are not opposites (zero-sum violation)"

    # --- Linear programming formulation for the scammer (Equation 6) ---
    # Decision variables: [v, x1, x2, ..., x10]
    # Objective: maximise v (equivalently, minimise -v)
    c = np.concatenate([-np.ones(1), np.zeros(10)])

    # Inequality constraints: v <= sum_i a_ij * x_i for each baiter strategy j
    # Rearranged: v - sum_i a_ij * x_i <= 0
    A_ub = np.column_stack([np.ones((10, 1)), -player1_matrix.T])
    b_ub = np.zeros(10)

    # Equality constraint: sum_i x_i = 1 (probability simplex)
    A_eq = np.concatenate([np.zeros(1), np.ones(10)]).reshape(1, 11)
    b_eq = np.ones(1)

    # Solve using the HiGHS solver
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  method="highs")

    # Extract results
    game_value = res.x[0]              # v* = game value
    player1_strategy = res.x[1:]        # Scammer's equilibrium strategy x*

    # Baiter's equilibrium derived from slack variables (dual relationship)
    player2_strategy = np.maximum(res.slack, 0)
    player2_strategy /= np.sum(player2_strategy)  # Normalise to probabilities

    # Save results
    save_game_results_to_json(
        game_value, player1_strategy, player2_strategy, filename=output_file
    )


if __name__ == "__main__":
    # NOTE: Update these paths to your local environment before running.
    folder_path = "data/deepseek/output"

    player1_file = os.path.join(folder_path, "average_payoff_matrix.csv")
    player2_file = os.path.join(folder_path, "negative_average_payoff_matrix.csv")
    output_file = os.path.join(folder_path, "nash_equilibrium_result.json")

    # Compute Nash equilibrium
    nash(player1_file, player2_file, output_file)

    # Optionally replace numerical strategy IDs with descriptive names
    # output_file_labelled = os.path.join(folder_path, "nash_equilibrium_labelled.json")
    # replace_strategy_names(output_file, output_file_labelled)
