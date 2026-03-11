"""
BotWarsMixtral.py

Main game orchestrator for Mixtral architecture simulations.

Coordinates the five-component pipeline for each turn (Figure 1).
Mixtral exhibits the most uniform baiter strategy distribution and
an 88% scammer win rate (Section 5.1).

Includes API rate limiting (5 calls/minute) to comply with
OctoAI API usage constraints.

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import json
import time
import datetime

from ActionBotMixtral import ActionBotMixtral
from ResponseBotMixtral import ResponseBotMixtral
from ResponseDataAnalyser import ResponseAnalyser
from SystemPromptGenerator import *
from UtilityCalculator import UtilityCalculator
from PayoffMatrixGeneratorFinal import generate_payoff_files


class BotWarsMixtral:
    """
    Orchestrates a complete Bot Wars game between Mixtral scammer and baiter.

    Functionally identical to BotWars (GPT), but uses Mixtral-8x22B-Instruct
    for strategy selection and message generation via OctoAI, with rate
    limiting to manage API call frequency.

    Attributes:
        scammer_R_bot: Scammer message generator (ResponseBotMixtral).
        victim_R_bot: Baiter message generator (ResponseBotMixtral).
        scammer_A_bot: Scammer strategy selector (ActionBotMixtral).
        victim_A_bot: Baiter strategy selector (ActionBotMixtral).
        utility_calculator: Computes turn-level utility (Equation 1).
        analyser: Extracts PII from dialogue (ResponseAnalyser).
    """

    def __init__(self):
        """Initialise all bots, utility calculator, and game state."""
        self.scammer_R_bot = ResponseBotMixtral(
            "scammer", get_scammer_response_prompt_mixtral()
        )
        self.victim_R_bot = ResponseBotMixtral(
            "victim", get_victim_response_prompt_mixtral()
        )
        self.scammer_A_bot = ActionBotMixtral(
            "scammer", get_scammer_action_prompt_mixtral()
        )
        self.victim_A_bot = ActionBotMixtral(
            "victim", get_victim_action_prompt_mixtral()
        )

        self.utility_calculator = UtilityCalculator()
        self.analyser = ResponseAnalyser(get_data_analysis_prompt())

        # Game state
        self.scammer_score = 0
        self.victim_score = 0
        self.turn = 0
        self.hang_up = False

        # Initial conditions
        self.victim_R_bot.response = "Who's on the line?"
        self.victim_A_bot.action = "Verification Request"
        self.scammer_R_bot.response = ""
        self.scammer_A_bot.action = ""

        # Save initial state
        self.victim_R_bot.save_opponent_response("")
        self.victim_A_bot.save_opponent_action("", 0)
        self.victim_A_bot.save_action(0)
        self.victim_R_bot.save_response()
        self.scammer_R_bot.save_opponent_response(self.victim_R_bot.response)
        self.scammer_A_bot.save_opponent_action(self.victim_A_bot.action, 0)

        # Game log
        self.game_log_path = self.create_log_filename()
        self.init_game_log()

    def create_log_filename(self):
        """Generate a unique log filename based on the current timestamp."""
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        return f"data/mixtral/game_progress_{timestamp}.json"

    def init_game_log(self):
        """Create an empty JSON array as the initial game log."""
        with open(self.game_log_path, "w") as file:
            json.dump([], file)

    def append_to_log(self, data):
        """Append a structured event to the game log."""
        with open(self.game_log_path, "r+") as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)

    def play(self, Tmax=50):
        """
        Execute the main game loop with API rate limiting.

        Args:
            Tmax (int): Maximum number of turns. Default: 50.
        """
        max_calls_per_minute = 5
        delay_between_calls = 60 / max_calls_per_minute

        while self.turn < Tmax:
            # Scammer turn with rate limiting
            self.scammer_A_bot.action = self.scammer_A_bot.generate_action(
                self.victim_R_bot.response,
                self.scammer_R_bot.response
            )
            time.sleep(delay_between_calls)
            self.scammer_R_bot.response = self.scammer_R_bot.generate_response(
                self.scammer_A_bot.action
            )
            print("scammer Action: ", self.scammer_A_bot.action)
            print("scammer: ", self.scammer_R_bot.response)

            # Baiter turn with rate limiting
            self.victim_A_bot.action = self.victim_A_bot.generate_action(
                self.scammer_R_bot.response,
                self.victim_R_bot.response,
                self.scammer_A_bot.action
            )
            time.sleep(delay_between_calls)
            self.victim_R_bot.response = self.victim_R_bot.generate_response(
                self.victim_A_bot.action
            )
            print("victim Action: ", self.victim_A_bot.action)
            print("victim: ", self.victim_R_bot.response)

            # Dialogue analysis and utility computation
            data = self.analyser.generate_data(
                self.scammer_R_bot.response, self.victim_R_bot.response
            )
            if self.scammer_A_bot.action == "Hang Up":
                self.hang_up = True
                print("hang-up")

            payoff = self.utility_calculator.utility(
                data, self.turn, self.hang_up
            )
            print("pay-off", payoff)
            self.scammer_score += payoff
            self.victim_score -= payoff

            # Save state
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

            # Log all events including detailed utility breakdown
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
            self.append_to_log({
                "type": "utility",
                "content": {
                    "payoff": payoff,
                    "scammer_score": self.scammer_score,
                    "victim_score": self.victim_score,
                    "turn": self.turn,
                    "full PII": self.utility_calculator.check_full_pii(),
                    "name": self.utility_calculator.pii["nameOnBankCard"],
                    "expiry": self.utility_calculator.pii["expiryDate"],
                    "card number": self.utility_calculator.pii["cardNumber"],
                    "pin": self.utility_calculator.pii["pin"],
                    "other PII": len(
                        self.utility_calculator.pii["otherPII"]
                    ),
                }
            })

            # Check termination conditions
            self.turn += 1
            print("turn:", self.turn)
            if self.hang_up or self.utility_calculator.check_full_pii():
                print("Game Over")
                self.scammer_A_bot.save_action_history()
                self.victim_A_bot.save_action_history()
                break

        # Persist action histories for cross-game learning
        self.scammer_A_bot.save_action_history()
        self.victim_A_bot.save_action_history()


if __name__ == "__main__":
    # Run 100 game simulations
    for i in range(100):
        game = BotWarsMixtral()
        game.play(50)

    # Generate per-game payoff matrices
    folder_path = "data/mixtral"
    generate_payoff_files(folder_path)
