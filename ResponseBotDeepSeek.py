"""
ResponseBotDeepSeek.py

Message generator bot for DeepSeek architecture. Translates a selected
strategy into contextually appropriate dialogue for either the scammer
or baiter role.

Uses the DeepSeek-Chat API (16B parameters) via an OpenAI-compatible
endpoint. Temperature is set to 0.7 (Section 4.1).

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import os
from datetime import datetime
from openai import OpenAI
from SystemPromptGenerator import *


class ResponseBotDeepSeek:
    """
    DeepSeek-based message generator for the Bot Wars framework.

    Generates dialogue that strictly adheres to the strategy selected
    by ActionBotDeepSeek. Uses a two-layer prompt architecture
    (Appendix C.2) to separate persona from strategy enactment.

    Attributes:
        role (str): Player role, either 'scammer' or 'victim' (baiter).
        model (str): DeepSeek model identifier.
        temperature (float): Sampling temperature (0.7 for DeepSeek).
        history (list): Conversation history for contextual coherence.
        response (str): Most recently generated message.
    """

    # Available models (listed for reference)
    models_list = [
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4",
    ]

    def __init__(self, role, system_prompt, model="deepseek-chat",
                 temperature=0.7, score=0):
        """
        Initialise the DeepSeek message generator.

        Args:
            role (str): 'scammer' or 'victim' (baiter).
            system_prompt (str): Role-specific message generator prompt.
            model (str): DeepSeek model identifier. Default: deepseek-chat.
            temperature (float): Sampling temperature. Default: 0.7.
            score (int): Initial cumulative score.
        """
        self.model = model
        # API key loaded from environment variable for security.
        # Set DEEPSEEK_API_KEY in your environment before running.
        self.APIKEY = os.environ.get("DEEPSEEK_API_KEY", "")
        self.temperature = temperature
        self.client = OpenAI(
            api_key=self.APIKEY,
            base_url="https://api.deepseek.com/v1"
        )
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
            action (str): The strategy to enact (e.g., 'Counter Questioning').
            opponent_response (str): The opponent's most recent message.

        Returns:
            str: The generated dialogue message (max 150 tokens).
        """
        messages = list(self.get_history())

        if opponent_response != "":
            messages.append({
                "role": "user",
                "content": f"{opponent_response}",
            })
        messages.append({
            "role": "user",
            "content": (f"generate the response strictly adhering to and "
                        f"aligning with the Action: {action}"),
        })

        try:
            answer = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=self.temperature,
                stream=False,
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
        return "DeepSeek"

    def get_role(self):
        """Return the bot's assigned role."""
        return self.role


if __name__ == "__main__":
    scammer_prompt = get_scammer_response_prompt_mixtral()
    scammer_bot = ResponseBotDeepSeek("scammer", scammer_prompt)

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
