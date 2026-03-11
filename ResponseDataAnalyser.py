"""
ResponseDataAnalyser.py

Dialogue analyser component that extracts PII from baiter responses.
This module implements the Dialogue Analyzer in the Bot Wars framework
(see Figure 1 in the paper).

At each turn, the analyser receives the scammer's message and the baiter's
response, then extracts any personally identifiable information (PII)
disclosed by the baiter. Extracted data is categorised into financial PII
(card name, number, expiry, PIN) and other PII (address, phone, etc.).

The extraction results are passed to the UtilityCalculator to compute
turn-level payoffs according to Equation 1 in the paper.

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import os
import json
from datetime import datetime
from openai import OpenAI
from SystemPromptGenerator import *


class ResponseAnalyser:
    """
    GPT-based dialogue analyser for PII extraction.

    Uses a lightweight GPT model (gpt-3.5-turbo by default) to analyse
    each turn's dialogue and extract structured PII data. The analyser
    operates on baiter responses only, using the scammer's message as
    context for understanding what information was requested.

    Manual validation on a randomly sampled 10% of conversations confirms
    a PII extraction accuracy of 94% (Section 4.2).

    Attributes:
        model (str): GPT model identifier for PII extraction.
        temperature (float): Sampling temperature for extraction.
        prompt (list): System prompt defining the extraction task.
    """

    # Available models for the analyser component
    models_list = [
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4",
    ]

    def __init__(self, system_prompt, model="gpt-3.5-turbo", temperature=1):
        """
        Initialise the dialogue analyser.

        Args:
            system_prompt (str): PII extraction prompt from
                SystemPromptGenerator.get_data_analysis_prompt().
            model (str): Model for extraction. Default: gpt-3.5-turbo
                (sufficient accuracy at lower cost for structured extraction).
            temperature (float): Sampling temperature. Default: 1.
        """
        self.model = model
        self.temperature = temperature
        # API key loaded from environment variable for security.
        # Set OPENAI_API_KEY in your environment before running.
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
        self.prompt = [{"role": "system", "content": system_prompt}]

    def generate_data(self, scammer_utterance, victim_utterance):
        """
        Extract PII from a single turn's dialogue.

        Sends the scammer's message (as context) and the baiter's response
        to the LLM, which returns a structured JSON object containing any
        extracted PII fields.

        Args:
            scammer_utterance (str): The scammer's message this turn.
            victim_utterance (str): The baiter's response this turn.

        Returns:
            dict: Extracted PII with keys matching the response_analysis_scheme:
                - nameOnBankCard (str): Name on bank card, or empty string.
                - expiryDate (str): Card expiry date, or empty string.
                - cardNumber (str): Partial or full card number, or empty string.
                - pin (str): PIN number, or empty string.
                - otherPII (str): Any other PII disclosed, or empty string.
        """
        messages = list(self.prompt)
        messages.append({
            "role": "user",
            "content": f"Scammer Request :{scammer_utterance}"
        })
        messages.append({
            "role": "user",
            "content": f"Victim Response :{victim_utterance}"
        })

        try:
            answer = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=self.temperature,
                stream=False,
                response_format={"type": "json_object"},
            )
        except ConnectionError as e:
            return "connection error"

        response = answer.choices[0].message.content
        final_response = self.transform_gpt_response_to_dict(response)
        return final_response

    def get_models(self):
        """Return the list of available model identifiers."""
        return self.models_list

    def get_model(self):
        """Return the currently configured model identifier."""
        return self.model

    def get_bot(self):
        """Return the bot architecture identifier."""
        return "GPT"

    def transform_gpt_response_to_dict(self, gpt_response):
        """
        Parse the LLM's JSON response into a Python dictionary.

        Args:
            gpt_response (str): Raw JSON string from the LLM.

        Returns:
            dict: Parsed PII data, or empty dict if parsing fails.
        """
        try:
            response_dict = json.loads(gpt_response)
            return response_dict
        except json.JSONDecodeError as e:
            print("Invalid JSON format:", e)
            return {}


if __name__ == "__main__":
    # Interactive mode for testing PII extraction
    systemPrompt = get_data_analysis_prompt()
    analysis_bot = ResponseAnalyser(systemPrompt)

    while True:
        scammer_prompt = input("scammer: ")
        victim_prompt = input("victim: ")
        T1 = datetime.now()
        response_analysis = analysis_bot.generate_data(
            scammer_prompt, victim_prompt
        )
        T2 = datetime.now()
        print("response: ", response_analysis)
        print("Generation Time: ", T2 - T1)
