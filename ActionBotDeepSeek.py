"""
ActionBotDeepSeek.py

Strategy selector bot for DeepSeek architecture. Selects strategic actions
(from 10 predefined strategies) for either the scammer or baiter role
based on conversation history and payoff feedback.

Uses the DeepSeek-Chat API (16B parameters) via an OpenAI-compatible
endpoint. Temperature is set to 0.7 (Section 4.1).

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


class ActionBotDeepSeek:
    """
    DeepSeek-based strategy selector for the Bot Wars framework.

    Implements the same strategy selection logic as ActionBotGPT but
    targets the DeepSeek-Chat API endpoint. DeepSeek achieves the strongest
    baiting performance (41% scammer win rate, 42-turn average duration)
    at substantially lower inference cost than GPT-4 (Section 5.1).

    Attributes:
        role (str): Player role, either 'scammer' or 'victim' (baiter).
        model (str): DeepSeek model identifier.
        temperature (float): Sampling temperature (0.7 for DeepSeek).
        history (list): Conversation history for in-context learning.
        actions (list): Available strategies based on assigned role.
        action (str): Most recently selected strategy.
    """

    # Available models (listed for reference; DeepSeek-Chat is the primary model)
    models_list = [
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4",
    ]

    # File paths for persisting action history across games.
    # NOTE: Update these paths to your local environment before running.
    scammer_history_file = "data/deepseek/scammer_action_history.json"
    victim_history_file = "data/deepseek/victim_action_history.json"

    def __init__(self, role, system_prompt, model="deepseek-chat",
                 temperature=0.7, score=0):
        """
        Initialise the DeepSeek strategy selector.

        Args:
            role (str): 'scammer' or 'victim' (baiter).
            system_prompt (str): Role-specific strategy selector prompt.
            model (str): DeepSeek model identifier. Default: deepseek-chat.
            temperature (float): Sampling temperature. Default: 0.7 (Section 4.1).
            score (int): Initial cumulative score.
        """
        # API key loaded from environment variable for security.
        # Set DEEPSEEK_API_KEY in your environment before running.
        self.APIKEY = os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.temperature = temperature
        # DeepSeek uses an OpenAI-compatible API endpoint
        self.client = OpenAI(
            api_key=self.APIKEY,
            base_url="https://api.deepseek.com/v1"
        )
        self.role = role
        self.score = score
        self.history = [{"role": "system", "content": system_prompt}]
        self.load_history_data()

        # Scammer strategies S1-S10 (Table 1)
        self.scammer_actions = [
            "Urgency",
            "Authority",
            "Emotional Manipulation",
            "Incentive",
            "Information Extraction",
            "Technical Jargon",
            "Reassurance",
            "Building Rapport",
            "Hang Up",
            "Persistence",
        ]

        # Baiter strategies B1-B10 (Table 1)
        self.baiter_actions = [
            "Delay",
            "Obfuscation",
            "Technical Difficulty",
            "Counter Questioning",
            "Information Fabrication",
            "Malicious Compliance",
            "Conversation Diversion",
            "Pretended Naivety",
            "Verification Request",
            "Reverse Engineering",
        ]

        self.actions = (self.baiter_actions if role == "victim"
                        else self.scammer_actions)
        self.action = ""

    def get_history(self):
        """Return the full interaction history including system prompt."""
        return self.history

    def generate_action(self, opponent_response, previous_response="",
                        opponent_action=""):
        """
        Generate a strategic action using DeepSeek-Chat.

        Args:
            opponent_response (str): The opponent's most recent message.
            previous_response (str): This player's most recent message.
            opponent_action (str): The opponent's most recent strategy.

        Returns:
            str: The selected strategy name. Falls back to random if parsing fails.
        """
        messages = list(self.get_history())

        if opponent_action != "":
            messages.append({
                "role": "user",
                "content": f"Opponent Action :{opponent_action}"
            })
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
            match = re.search(r'"Strategy":\s*"([^"]+)"', response)
            action = match.group(1) if match else None
            final_action = (action if action in self.actions
                            else random.choice(self.actions))
            if action != final_action:
                print("random action")
            self.action = final_action
        except ConnectionError as e:
            return "connection error"
        return final_action

    def save_action(self, score=0, data_extracted=""):
        """Record this player's selected action and resulting payoff."""
        history_entry = {
            "role": "assistant",
            "content": f"Action :{self.action}, Score: {score}"
        }
        if self.role == "victim":
            history_entry["content"] += f", Data: {data_extracted}"
        self.history.append(history_entry)

    def save_opponent_action(self, reply, score, data_extracted=""):
        """Record the opponent's action and resulting payoff."""
        history_entry = {
            "role": "user",
            "content": f"Action :{reply}, Score: {score}"
        }
        if self.role != "victim":
            history_entry["content"] += f", Data: {data_extracted}"
        self.history.append(history_entry)

    def get_models(self):
        """Return the list of available model identifiers."""
        return self.models_list

    def get_model(self):
        """Return the currently configured model identifier."""
        return self.model

    def get_bot(self):
        """Return the bot architecture identifier."""
        return "DeepSeek"

    def get_role(self):
        """Return the bot's assigned role."""
        return self.role

    def save_action_history(self):
        """
        Persist the current game's action history to disk.
        Maintains a rolling window of the 200 most recent entries.
        """
        file_path = (self.scammer_history_file if self.role == "scammer"
                     else self.victim_history_file)
        current_history = self.history[1:]  # Exclude system prompt

        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                file_data = json.load(file)
            file_data.extend(current_history)
            file_data = file_data[-200:]
            with open(file_path, "w") as file:
                json.dump(file_data, file, indent=4)
        else:
            with open(file_path, "w") as file:
                json.dump(current_history, file, indent=4)

        self.history = [{"role": "system", "content": self.history[0]["content"]}]

    def load_history_data(self):
        """Load action history from previous games for cross-game learning."""
        file_path = (self.victim_history_file if self.role == "victim"
                     else self.scammer_history_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    file_data = json.load(file)
                self.history.extend(file_data)
            except json.JSONDecodeError:
                pass


if __name__ == "__main__":
    scammer_prompt = get_scammer_action_prompt_deepseek()
    scammer_bot = ActionBotDeepSeek("scammer", scammer_prompt)
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
