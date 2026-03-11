"""
BotWarsGPT.py

Main game orchestrator for GPT-4 architecture simulations.

Coordinates the five-component pipeline for each turn (Figure 1):
  1. Scammer Strategy Selector -> selects scammer action
  2. Scammer Message Generator -> generates scammer dialogue
  3. Baiter Strategy Selector  -> selects baiter action
  4. Baiter Message Generator  -> generates baiter dialogue
  5. Dialogue Analyser         -> extracts PII and computes utility

Runs 100 games per architecture with a maximum of 50 turns each.
Games terminate upon complete FPII extraction, scammer hang-up,
or reaching t_max turns.

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import json
import datetime

from ActionBotGPT import ActionBotGPT
from ResponseBotGPT import ResponseBotGPT
from ResponseDataAnalyser import ResponseAnalyser
from SystemPromptGenerator import *
from UtilityCalculator import UtilityCalculator
from PayoffMatrixGeneratorFinal import generate_payoff_files


class BotWars:
    """
    Orchestrates a complete Bot Wars game between GPT-4 scammer and baiter.

    Each game consists of alternating turns where:
    - The scammer selects a strategy and generates a message
    - The baiter selects a counter-strategy and generates a response
    - The dialogue analyser extracts PII and computes utility

    The game terminates when complete FPII is extracted, the scammer
    hangs up, or the maximum turn count is reached (Section 4.2).

    Attributes:
        scammer_R_bot: Scammer message generator (ResponseBotGPT).
        victim_R_bot: Baiter message generator (ResponseBotGPT).
        scammer_A_bot: Scammer strategy selector (ActionBotGPT).
        victim_A_bot: Baiter strategy selector (ActionBotGPT).
        utility_calculator: Computes turn-level utility (Equation 1).
        analyser: Extracts PII from dialogue (ResponseAnalyser).
        scammer_score: Cumulative scammer utility across all turns.
        victim_score: Cumulative baiter utility (negative of scammer).
        turn: Current turn index.
        hang_up: Whether the scammer has selected S10 (Hang Up).
    """

    def __init__(self):
        """Initialise all bots, utility calculator, and game state."""
        # Initialise the four LLM-based bots with role-specific prompts
        self.scammer_R_bot = ResponseBotGPT(
            "scammer", get_scammer_response_prompt()
        )
        self.victim_R_bot = ResponseBotGPT(
            "victim", get_victim_response_prompt()
        )
        self.scammer_A_bot = ActionBotGPT(
            "scammer", get_scammer_action_prompt()
        )
        self.victim_A_bot = ActionBotGPT(
            "victim", get_victim_action_prompt()
        )

        # Initialise utility computation and dialogue analysis
        self.utility_calculator = UtilityCalculator()
        self.analyser = ResponseAnalyser(get_data_analysis_prompt())

        # Initialise game state
        self.scammer_score = 0
        self.victim_score = 0
        self.turn = 0
        self.hang_up = False

        # Set initial conditions: baiter opens with a verification request
        self.victim_R_bot.response = "Who's on the line?"
        self.victim_A_bot.action = "Verification Request"
        self.scammer_R_bot.response = ""
        self.scammer_A_bot.action = ""

        # Save initial state to conversation histories
        self.victim_R_bot.save_opponent_response("")
        self.victim_A_bot.save_opponent_action("", 0)
        self.victim_A_bot.save_action(0)
        self.victim_R_bot.save_response()
        self.scammer_R_bot.save_opponent_response(self.victim_R_bot.response)
        self.scammer_A_bot.save_opponent_action(self.victim_A_bot.action, 0)

        # Initialise game log file
        self.game_log_path = self.create_log_filename()
        self.init_game_log()

    def create_log_filename(self):
        """
        Generate a unique log filename based on the current timestamp.

        Returns:
            str: Path to the game log JSON file.
        """
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        return f"data/gpt/game_progress_{timestamp}.json"

    def init_game_log(self):
        """Create an empty JSON array as the initial game log."""
        with open(self.game_log_path, "w") as file:
            json.dump([], file)

    def append_to_log(self, data):
        """
        Append a structured event to the game log.

        Args:
            data (dict): Event data with 'type' and 'content' keys.
                Types include: scammer_action, scammer_response,
                victim_action, victim_response, data_extracted, utility.
        """
        with open(self.game_log_path, "r+") as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)

    def play(self, Tmax=50):
        """
        Execute the main game loop.

        Each iteration represents one complete turn comprising five
        sequential LLM calls (Figure 1). The loop continues until
        one of three termination conditions is met.

        Args:
            Tmax (int): Maximum number of turns. Default: 50 (Section 3.1).
        """
        while self.turn < Tmax:
            # --- Scammer's turn ---
            # Step 1: Scammer strategy selector chooses action
            self.scammer_A_bot.action = self.scammer_A_bot.generate_action(
                self.victim_R_bot.response, self.scammer_R_bot.response
            )
            # Step 2: Scammer message generator produces dialogue
            self.scammer_R_bot.response = self.scammer_R_bot.generate_response(
                self.scammer_A_bot.action
            )

            # --- Baiter's turn ---
            # Step 3: Baiter strategy selector chooses counter-action
            self.victim_A_bot.action = self.victim_A_bot.generate_action(
                self.scammer_R_bot.response,
                self.victim_R_bot.response,
                self.scammer_A_bot.action,
            )
            # Step 4: Baiter message generator produces response
            self.victim_R_bot.response = self.victim_R_bot.generate_response(
                self.victim_A_bot.action, self.scammer_R_bot.response
            )

            # --- Analysis ---
            # Step 5: Dialogue analyser extracts PII and computes utility
            data = self.analyser.generate_data(
                self.scammer_R_bot.response, self.victim_R_bot.response
            )

            # Compute turn payoff (Equation 1)
            payoff = self.utility_calculator.utility(
                data, self.turn, self.hang_up
            )
            self.scammer_score += payoff
            self.victim_score -= payoff  # Zero-sum: U_B = -U_S

            # --- State persistence ---
            # Save actions and responses to bot histories
            self.scammer_A_bot.save_action(payoff)
            self.scammer_R_bot.save_response()
            self.scammer_R_bot.save_opponent_response(
                self.victim_R_bot.response
            )
            self.scammer_A_bot.save_opponent_action(
                self.victim_A_bot.action, -payoff, data
            )
            self.victim_R_bot.save_opponent_response(
                self.scammer_R_bot.response
            )
            self.victim_A_bot.save_opponent_action(
                self.scammer_A_bot.action, payoff
            )
            self.victim_A_bot.save_action(-payoff, data)
            self.victim_R_bot.save_response()

            # Log all events for this turn
            self.append_to_log({
                "type": "scammer_action",
                "content": self.scammer_A_bot.action
            })
            self.append_to_log({
                "type": "scammer_response",
                "content": self.scammer_R_bot.response
            })
            self.append_to_log({
                "type": "victim_action",
                "content": self.victim_A_bot.action
            })
            self.append_to_log({
                "type": "victim_response",
                "content": self.victim_R_bot.response
            })
            self.append_to_log({
                "type": "data_extracted",
                "content": data
            })

            # Advance turn counter and check termination conditions
            self.turn += 1
            if self.hang_up or self.utility_calculator.check_full_pii():
                break

        # Persist action histories for cross-game learning
        self.scammer_A_bot.save_action_history()
        self.victim_A_bot.save_action_history()


if __name__ == "__main__":
    # Run 100 game simulations (Section 4.2: 100 games per architecture)
    for i in range(100):
        game = BotWars()
        game.play(50)

    # Generate per-game payoff matrices for Nash equilibrium computation
    folder_path = "data/gpt"
    generate_payoff_files(folder_path)
