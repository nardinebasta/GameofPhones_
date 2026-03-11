"""
ActionBotGPT.py

Strategy selector bot for GPT-4 architecture. Selects strategic actions
(from 10 predefined strategies) for either the scammer or baiter role
based on conversation history and payoff feedback.

This module implements the Strategy Selector component of the Bot Wars
framework (see Figure 1 in the paper). At each turn, it analyses the
history of actions, opponent responses, and cumulative payoffs to select
the next strategy from the player's action space.

The selected strategy is then passed to the corresponding ResponseBot
(message generator) to produce contextually appropriate dialogue.

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import os
import re
import json
import random
from datetime import datetime
from openai import OpenAI
from SystemPromptGenerator import *


class ActionBotGPT:
    """
    GPT-4 based strategy selector for the Bot Wars framework.

    This bot maintains a history of selected strategies and their payoffs,
    using this experience to inform future strategy selections via
    in-context learning. The bot outputs structured JSON containing the
    selected strategy, which is parsed and validated against the
    predefined action space.

    Attributes:
        role (str): Player role, either 'scammer' or 'victim' (baiter).
        model (str): GPT model identifier for API calls.
        temperature (float): Controls randomness in strategy selection.
        history (list): Conversation history including system prompt,
            past actions, and payoff outcomes.
        actions (list): Available strategies based on the assigned role.
        action (str): Most recently selected strategy.
    """

    # Available GPT model versions
    models_list = [
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4",
    ]

    # File paths for persisting action history across games.
    # The history from recent games is loaded as context for strategy
    # selection, enabling learning across interactions (Section 4.4).
    # NOTE: Update these paths to your local environment before running.
    scammer_history_file = "data/gpt/scammer_action_history.json"
    victim_history_file = "data/gpt/victim_action_history.json"

    def __init__(self, role, system_prompt, model="gpt-4-0125-preview",
                 temperature=1, score=0):
        """
        Initialise the GPT-4 strategy selector.

        Args:
            role (str): 'scammer' or 'victim' (baiter).
            system_prompt (str): Role-specific strategy selector prompt
                (see Appendix C.1 in Supplementary Material).
            model (str): GPT model identifier. Default: gpt-4-0125-preview.
            temperature (float): Sampling temperature. Default: 1.0 for GPT-4
                (calibrated through preliminary trials, Section 4.1).
            score (int): Initial cumulative score.
        """
        self.model = model
        self.temperature = temperature
        # API key loaded from environment variable for security.
        # Set OPENAI_API_KEY in your environment before running.
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.role = role
        self.score = score
        # Initialise history with the strategy selector system prompt
        self.history = [{"role": "system", "content": system_prompt}]
        # Load action history from previous games to enable cross-game learning
        self.load_history_data()

        # Define the 10 scammer strategies (S1-S10) from Table 1
        self.scammer_actions = [
            "Urgency",               # S1: Artificial time constraints
            "Authority",              # S2: False institutional power
            "Emotional Manipulation", # S3: Evoke emotional responses
            "Incentive",              # S4: Present rewards
            "Information Extraction", # S5: Acquire PII
            "Technical Jargon",       # S6: Impair critical thinking
            "Reassurance",            # S7: Fabricated guarantees
            "Building Rapport",       # S8: Cultivate connection
            "Hang Up",               # S10: Terminate conversation
            "Persistence",            # S9: Continuous pressure
        ]

        # Define the 10 baiter strategies (B1-B10) from Table 1
        self.baiter_actions = [
            "Delay",                  # B1: Prolong interaction
            "Obfuscation",           # B2: Unclear responses
            "Technical Difficulty",   # B3: Artificial obstacles
            "Counter Questioning",    # B4: Challenge narrative
            "Information Fabrication",# B5: Provide false data
            "Malicious Compliance",   # B6: Misconstrue instructions
            "Conversation Diversion", # B7: Disrupt scripts
            "Pretended Naivety",      # B8: Feign misunderstanding
            "Verification Request",   # B9: Challenge legitimacy
            "Reverse Engineering",    # B10: Extract scammer info
        ]

        # Assign action space based on role
        self.actions = (self.baiter_actions if role == "victim"
                        else self.scammer_actions)
        self.action = ""

    def get_history(self):
        """Return the full interaction history including system prompt."""
        return self.history

    def generate_action(self, opponent_response, previous_response="",
                        opponent_action=""):
        """
        Generate a strategic action based on the current game state.

        Constructs a prompt from the conversation history, the opponent's
        last action and response, and the player's own last response. The
        LLM then selects a strategy via chain-of-thought reasoning over
        the accumulated payoff data (Section 4.4).

        Args:
            opponent_response (str): The opponent's most recent message.
            previous_response (str): This player's most recent message.
            opponent_action (str): The opponent's most recent strategy.

        Returns:
            str: The selected strategy name from the action space.
                 Falls back to random selection if parsing fails.
        """
        messages = list(self.get_history())

        # Provide the opponent's action for context if available
        if opponent_action != "":
            messages.append({
                "role": "user",
                "content": f"Opponent Action :{opponent_action}"
            })
        # Provide conversation context
        messages.append({
            "role": "user",
            "content": f"Last Response :{previous_response}"
        })
        messages.append({
            "role": "user",
            "content": f"Opponent Response :{opponent_response}"
        })
        messages.append({
            "role": "user",
            "content": "Generate an action in response to the opponent"
        })

        # Call the GPT-4 API with JSON response format for structured output
        try:
            answer = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=self.temperature,
                stream=False,
            )
            response = answer.choices[0].message.content
            # Parse the strategy from the JSON response
            match = re.search(r'"Strategy":\s*"([^"]+)"', response)
            action = match.group(1) if match else None
            # Validate against the action space; fall back to random if invalid
            final_action = (action if action in self.actions
                            else random.choice(self.actions))
            if action != final_action:
                print("random action")
            self.action = final_action
        except ConnectionError as e:
            return "connection error"
        return final_action

    def save_action(self, score=0, data_extracted=""):
        """
        Record this player's selected action and resulting payoff.

        Args:
            score (float): The payoff received for this turn (Equation 1).
            data_extracted (str): PII extracted this turn (baiter role only).
        """
        entry_content = f"Action :{self.action}, Score: {score}"
        if self.role == "victim":
            entry_content += f", Data: {data_extracted}"
        self.history.append({"role": "assistant", "content": entry_content})

    def save_opponent_action(self, reply, score, data_extracted=""):
        """
        Record the opponent's action and resulting payoff.

        Args:
            reply (str): The opponent's selected strategy.
            score (float): The payoff from the opponent's perspective.
            data_extracted (str): PII extracted (scammer role only).
        """
        entry_content = f"Action :{reply}, Score: {score}"
        if self.role != "victim":
            entry_content += f", Data: {data_extracted}"
        self.history.append({"role": "user", "content": entry_content})

    def get_models(self):
        """Return the list of available model identifiers."""
        return self.models_list

    def get_model(self):
        """Return the currently configured model identifier."""
        return self.model

    def get_bot(self):
        """Return the bot architecture identifier."""
        return "GPT"

    def get_role(self):
        """Return the bot's assigned role."""
        return self.role

    def save_action_history(self):
        """
        Persist the current game's action history to disk.

        Appends the current game's actions to the history file, maintaining
        a rolling window of the 200 most recent entries. This stored history
        is loaded at the start of subsequent games to provide cross-game
        context for strategy selection (Section 4.4: 20 most recent games).
        """
        file_path = (self.scammer_history_file if self.role == "scammer"
                     else self.victim_history_file)
        current_history = self.history[1:]  # Exclude system prompt

        file_data = []
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                file_data = json.load(file)
            file_data.extend(current_history)
            file_data = file_data[-200:]  # Keep rolling window of 200 entries
            with open(file_path, "w") as file:
                json.dump(file_data, file, indent=4)
        else:
            with open(file_path, "w") as file:
                json.dump(current_history, file, indent=4)

        # Reset history to system prompt for next game
        self.history = [{"role": "system", "content": self.history[0]["content"]}]

    def load_history_data(self):
        """
        Load action history from previous games.

        Loads persisted action history to provide cross-game context
        for strategy selection, enabling in-context learning across
        repeated interactions (Section 4.4).
        """
        file_path = (self.victim_history_file if self.role == "victim"
                     else self.scammer_history_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    file_data = json.load(file)
                self.history.extend(file_data)
            except json.JSONDecodeError:
                pass  # Handle empty or corrupted file gracefully


if __name__ == "__main__":
    # Example: initialise a scammer strategy selector and run interactively
    scammer_prompt = get_scammer_action_prompt()
    scammer_bot = ActionBotGPT("scammer", scammer_prompt)
    print(scammer_bot.actions)

    while True:
        victim_response = input("victim: ")
        T1 = datetime.now()
        scammer_response = scammer_bot.generate_action(victim_response)
        scammer_bot.save_opponent_action(victim_response, 0)
        scammer_bot.save_action(0)
        T2 = datetime.now()
        print("Scammer: ", scammer_response)
        print("Generation Time: ", T2 - T1)
        scammer_bot.save_action_history()
