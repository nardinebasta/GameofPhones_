"""
PayoffMatrixGeneratorFinal.py

Constructs empirical payoff matrices from simulated game logs.

This module implements the payoff matrix construction described in
Section 3.4 (Equation 5). For each game log, it computes the average
scammer payoff for every (S_i, B_j) strategy pair observed, producing
a 10x10 payoff matrix. These per-game matrices are then averaged across
all games to yield the final empirical payoff matrix used for Nash
equilibrium computation.

The module also generates the negated payoff matrix required for the
linear programming formulation of the zero-sum game (Section 3.4).

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd


# =============================================================================
# Strategy Definitions (Table 1)
# =============================================================================

# Baiter strategies B1-B10 (column labels in the payoff matrix)
VICTIM_ACTIONS = [
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

# Scammer strategies S1-S10 (row labels in the payoff matrix)
SCAMMER_ACTIONS = [
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


def generate_payoff_files(folder_path):
    """
    Generate per-game payoff matrices from game log JSON files.

    For each game log in folder_path, extracts all (scammer_action,
    victim_action, payoff) triples and computes the average payoff
    for each strategy pair using Equation 5:

        a_ij = sum(U_S(t) * I(i,j,t)) / sum(I(i,j,t))

    Each resulting matrix is saved as a CSV file alongside the
    original JSON log.

    Args:
        folder_path (str): Directory containing game log JSON files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)

            # Initialise 10x10 matrices for payoff sums and counts
            payoff_matrix = np.zeros((10, 10))
            count_matrix = np.zeros((10, 10))

            # Extract (scammer_action, victim_action, payoff) triples
            actions = []
            for entry in data:
                if entry["type"] == "scammer_action":
                    scammer_action = entry["content"]
                elif entry["type"] == "victim_action":
                    victim_action = entry["content"]
                elif entry["type"] == "utility":
                    payoff = entry["content"]["payoff"]
                    actions.append((scammer_action, victim_action, payoff))
            print(actions)

            # Populate the payoff matrix with cumulative payoffs
            for scammer_action, victim_action, payoff in actions:
                x = SCAMMER_ACTIONS.index(scammer_action)
                y = VICTIM_ACTIONS.index(victim_action)
                payoff_matrix[x, y] += payoff
                count_matrix[x, y] += 1

            print(payoff_matrix)
            print(count_matrix)

            # Compute average payoff per strategy pair (Equation 5)
            with np.errstate(divide="ignore", invalid="ignore"):
                average_payoff_matrix = np.divide(
                    payoff_matrix, count_matrix, where=count_matrix != 0
                )
            print(average_payoff_matrix)

            # Save as labelled CSV
            df = pd.DataFrame(
                average_payoff_matrix,
                index=SCAMMER_ACTIONS,
                columns=VICTIM_ACTIONS
            )
            print(df)
            csv_filename = os.path.splitext(filename)[0] + "_payoff_matrix.csv"
            df.to_csv(os.path.join(folder_path, csv_filename))
            print(f"Processed {filename} into {csv_filename}")


def generate_payoff_files2(folder_path):
    """
    Alternative payoff matrix generator (same logic as generate_payoff_files).

    Processes game data from JSON files to compute individual game
    payoff matrices. Provided for backward compatibility.

    Args:
        folder_path (str): Directory containing JSON game log files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)

            payoff_matrix = np.zeros((10, 10))
            count_matrix = np.zeros((10, 10))

            actions = []
            for entry in data:
                if entry["type"] == "scammer_action":
                    scammer_action = entry["content"]
                elif entry["type"] == "victim_action":
                    victim_action = entry["content"]
                elif entry["type"] == "utility":
                    payoff = entry["content"]["payoff"]
                    actions.append((scammer_action, victim_action, payoff))

            for scammer_action, victim_action, payoff in actions:
                x = SCAMMER_ACTIONS.index(scammer_action)
                y = VICTIM_ACTIONS.index(victim_action)
                payoff_matrix[x, y] += payoff
                count_matrix[x, y] += 1

            with np.errstate(divide="ignore", invalid="ignore"):
                average_payoff_matrix = np.divide(
                    payoff_matrix, count_matrix, where=count_matrix != 0
                )

            df = pd.DataFrame(
                average_payoff_matrix,
                index=SCAMMER_ACTIONS,
                columns=VICTIM_ACTIONS
            )
            csv_filename = os.path.splitext(filename)[0] + "_payoff_matrix.csv"
            df.to_csv(os.path.join(folder_path, csv_filename))


def compute_average_payoff_matrix(input_folder):
    """
    Compute the elementwise average payoff matrix across all games.

    Reads all per-game payoff matrix CSVs, verifies consistent formatting,
    and computes the elementwise mean. This produces the final empirical
    payoff matrix A = [a_ij] used for Nash equilibrium computation.

    Args:
        input_folder (str): Path to the folder containing per-game
            payoff matrix CSV files.

    Returns:
        pd.DataFrame: The averaged 10x10 payoff matrix with strategy
            labels as row/column indices.

    Raises:
        ValueError: If no CSV files are found or matrices have
            inconsistent formats.
    """
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {input_folder}")

    matrices = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, index_col=0)
            matrices.append(df)
            print(f"Loaded '{file_path}' successfully.")
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}. Skipping.")

    if not matrices:
        raise ValueError("No valid payoff matrix files could be read.")

    # Verify all matrices have identical row/column structure
    base_df = matrices[0]
    base_rows = list(base_df.index)
    base_cols = list(base_df.columns)

    for i, df in enumerate(matrices[1:], start=2):
        if (list(df.index) != base_rows) or (list(df.columns) != base_cols):
            raise ValueError(
                f"Matrix in file #{i} has a different format than the base."
            )

    # Compute elementwise average
    avg_matrix = sum(matrices) / len(matrices)
    return avg_matrix


def average_payoff_matrix(input_folder, output_folder):
    """
    Compute and save the average payoff matrix to CSV.

    Args:
        input_folder (str): Folder containing per-game payoff CSVs.
        output_folder (str): Folder to save the averaged matrix.
    """
    try:
        avg_matrix = compute_average_payoff_matrix(input_folder)
    except Exception as e:
        print("Error computing average payoff matrix:", e)
        sys.exit(1)

    output_file = os.path.join(output_folder, "average_payoff_matrix.csv")
    try:
        avg_matrix.to_csv(output_file)
        print(f"Average payoff matrix saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving the average payoff matrix: {e}")
        sys.exit(1)


def negate_csv_values(input_filepath, output_filepath):
    """
    Negate all values in a payoff matrix CSV.

    Under the zero-sum assumption (Section 3.1), the baiter's payoff
    matrix is the negation of the scammer's. This function generates
    the negated matrix required for the linear programming formulation
    of Nash equilibrium (Equations 6-7).

    Args:
        input_filepath (str): Path to the scammer's payoff matrix CSV.
        output_filepath (str): Path to save the negated (baiter) matrix.
    """
    data = pd.read_csv(input_filepath, index_col=0)
    negative_data = -data
    negative_data.to_csv(output_filepath)


if __name__ == "__main__":
    # Example usage: process DeepSeek game logs
    # NOTE: Update these paths to your local environment before running.
    folder_path = "data/deepseek"
    generate_payoff_files(folder_path)

    input_folder = folder_path
    output_folder = "data/deepseek/output/"
    os.makedirs(output_folder, exist_ok=True)

    # Compute and save the average payoff matrix
    average_payoff_matrix(input_folder, output_folder)

    # Generate negated matrix for Nash equilibrium computation
    negate_csv_values(
        os.path.join(output_folder, "average_payoff_matrix.csv"),
        os.path.join(output_folder, "negative_average_payoff_matrix.csv"),
    )
