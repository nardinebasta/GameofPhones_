"""
ResponseBotMixtral.py

Message generator bot for Mixtral architecture. Translates a selected
strategy into contextually appropriate dialogue for either the scammer
or baiter role.

Uses the Mixtral-8x22B-Instruct model via the OctoAI API.
Temperature is set to 0.7 (Section 4.1).

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import os
from datetime import datetime
from octoai.client import OctoAI
from SystemPromptGenerator import *


class ResponseBotMixtral:
    """
    Mixtral-based message generator for the Bot Wars framework.

    Generates dialogue that strictly adheres to the strategy selected
    by ActionBotMixtral. Uses a two-layer prompt architecture
    (Appendix C.2) to separate persona from strategy enactment.

    Attributes:
        role (str): Player role, either 'scammer' or 'victim' (baiter).
        model (str): Mixtral model identifier.
        temperature (float): Sampling temperature (0.7 for Mixtral).
        history (list): Conversation history for contextual coherence.
        response (str): Most recently generated message.
    """

    # Available Mixtral model versions via OctoAI
    models_list = [
        "mixtral-8x22b-instruct",
        "mixtral-8x7b-instruct-fp16",
        "mistral-7b-instruct-fp16",
    ]

    # API key loaded from environment variable for security.
    # Set OCTOAI_API_KEY in your environment before running.
    OCTOAI_TOKEN = os.environ.get("OCTOAI_API_KEY", "")
    client = OctoAI(api_key=OCTOAI_TOKEN)

    def __init__(self, role, system_prompt, model="mixtral-8x22b-instruct",
                 temperature=0.7, score=0):
        """
        Initialise the Mixtral message generator.

        Args:
            role (str): 'scammer' or 'victim' (baiter).
            system_prompt (str): Role-specific message generator prompt.
            model (str): Mixtral model identifier.
            temperature (float): Sampling temperature. Default: 0.7.
            score (int): Initial cumulative score.
        """
        self.model = model
        self.temperature = temperature
        self.client = ResponseBotMixtral.client
        self.role = role
        self.score = score
        self.history = [{"role": "system", "content": system_prompt}]
        self.response = ""

    def get_history(self):
        """Return the full conversation history including system prompt."""
        return self.history

    def generate_response(self, action, opponent_response=""):
        """
        Generate a dialogue message that enacts the specified strategy.

        Args:
            action (str): The strategy to enact.
            opponent_response (str): The opponent's most recent message.

        Returns:
            str: The generated dialogue message (max 150 tokens).
        """
        messages = list(self.get_history())

        if opponent_response != "":
            messages.append({
                "role": "user",
                "content": f"{opponent_response}"
            })
        messages.append({
            "role": "user",
            "content": (f"generate the response strictly adhering to and "
                        f"aligning with the Action: {action}")
        })

        try:
            answer = self.client.text_gen.create_chat_completion(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=self.temperature,
            )
            self.response = answer.choices[0].message.content
        except ConnectionError as e:
            return "connection error"

        return self.response

    def save_response(self):
        """Append the generated response to the conversation history."""
        self.history.append({"role": "assistant", "content": self.response})

    def save_opponent_response(self, reply):
        """Append the opponent's response to the conversation history."""
        self.history.append({"role": "user", "content": reply})

    def get_models(self):
        """Return the list of available model identifiers."""
        return self.models_list

    def get_model(self):
        """Return the currently configured model identifier."""
        return self.model

    def get_bot(self):
        """Return the bot architecture identifier."""
        return "Mixtral"

    def get_role(self):
        """Return the bot's assigned role."""
        return self.role


if __name__ == "__main__":
    scammer_prompt = get_scammer_response_prompt()
    scammer_bot = ResponseBotMixtral("scammer", scammer_prompt)

    while True:
        victim_response = input("victim: ")
        T1 = datetime.now()
        scammer_response = scammer_bot.generate_response(
            "Urgency", victim_response
        )
        T2 = datetime.now()
        print("Scammer: ", scammer_response)
        print("Generation Time: ", T2 - T1)
        scammer_bot.save_opponent_response(victim_response)
        scammer_bot.save_response()
        print("History:", scammer_bot.get_history())
