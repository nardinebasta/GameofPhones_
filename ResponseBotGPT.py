"""
ResponseBotGPT.py

Message generator bot for GPT-4 architecture. Translates a selected
strategy into contextually appropriate dialogue for either the scammer
or baiter role.

This module implements the Message Generator component of the Bot Wars
framework (see Figure 1 in the paper). It receives a strategy label from
the corresponding ActionBot and generates a natural language message that
enacts that strategy within the ongoing conversation.

The prompt uses a two-layer architecture (Appendix C.2):
  - Base Context Layer: establishes persona characteristics
  - Behavioral Layer: implements the selected strategy as conversational tactics

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import os
from datetime import datetime
from openai import OpenAI
from SystemPromptGenerator import *


class ResponseBotGPT:
    """
    GPT-4 based message generator for the Bot Wars framework.

    Generates dialogue that strictly adheres to the strategy selected
    by the corresponding ActionBotGPT. Maintains conversation history
    for contextual coherence across turns.

    Attributes:
        role (str): Player role, either 'scammer' or 'victim' (baiter).
        model (str): GPT model identifier for API calls.
        temperature (float): Controls response creativity.
        history (list): Conversation history including system prompt.
        response (str): Most recently generated message.
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

    def __init__(self, role, system_prompt, model="gpt-4",
                 temperature=1, score=0):
        """
        Initialise the GPT-4 message generator.

        Args:
            role (str): 'scammer' or 'victim' (baiter).
            system_prompt (str): Role-specific message generator prompt
                (see Appendix C.2 in Supplementary Material).
            model (str): GPT model identifier. Default: gpt-4.
            temperature (float): Sampling temperature. Default: 1.0 for GPT-4.
            score (int): Initial cumulative score.
        """
        self.model = model
        self.temperature = temperature
        # API key loaded from environment variable for security.
        # Set OPENAI_API_KEY in your environment before running.
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
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

        Constructs a prompt from the conversation history and appends
        a directive to generate a message aligned with the selected
        strategy. The Behavioral Layer of the prompt translates the
        abstract strategy into concrete conversational tactics.

        Args:
            action (str): The strategy to enact (e.g., 'Delay', 'Urgency').
            opponent_response (str): The opponent's most recent message,
                providing conversational context.

        Returns:
            str: The generated dialogue message (max 150 tokens).
        """
        messages = list(self.get_history())

        # Append opponent's message for conversational context
        if opponent_response != "":
            messages.append({
                "role": "user",
                "content": f"{opponent_response}",
            })
        # Directive to generate response aligned with the selected strategy
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
        return "GPT"

    def get_role(self):
        """Return the bot's assigned role."""
        return self.role


if __name__ == "__main__":
    # Example: initialise a scammer message generator and run interactively
    scammer_prompt = get_scammer_response_prompt()
    scammer_bot = ResponseBotGPT("scammer", scammer_prompt)

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
