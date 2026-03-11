"""
UtilityCalculator.py

Computes the scammer's utility function (Equation 1 in the paper) at
each turn of the Bot Wars simulation. The utility captures the trade-off
between information extraction value and time expenditure.

Utility function:
    U_S(t) = I_f + w_o * I_o - w_t * (t / t_max) + W_s(t)

Where:
    I_f  = Financial PII extraction score (Equation 2)
    I_o  = Non-financial PII score with diminishing returns (Equation 3)
    w_t  = Time penalty weight
    W_s  = Termination payoffs (hang-up bonus or full PII bonus)

Parameters (Section 3.3):
    t_max = 50, w_o = 5, w_t = 5, k = 0.15
    W_hangup = 15, W_fullPII = 100

Reference:
    Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars'
    to Counteract Phone Scams (ECML-PKDD 2026)
"""

import math


class UtilityCalculator:
    """
    Computes turn-level utility scores for the Bot Wars framework.

    Tracks cumulative PII disclosure across turns and computes the
    scammer's utility at each turn based on the financial PII extraction
    rate, non-financial PII count, time penalty, and termination bonuses.

    Under the zero-sum structure, the baiter's utility is U_B(t) = -U_S(t).

    Attributes:
        pii (dict): Cumulative PII state across all turns.
            Keys: nameOnBankCard, expiryDate, cardNumber, pin, otherPII.
        turn_pii (dict): PII extracted in the current turn only.
        weights (dict): Utility function weights (w_f, w_o, w_t).
        bonuses (dict): Termination payoffs (W_hangup, W_fullPII).
        k (float): Saturation rate for non-financial PII (Equation 3).
    """

    def __init__(self):
        """
        Initialise the utility calculator with empty PII state and
        calibrated parameters from Section 3.3.
        """
        # Cumulative PII state tracking across all turns
        self.pii = {
            "nameOnBankCard": "",   # Cardholder name
            "expiryDate": "",       # Card expiry date
            "cardNumber": "",       # Card number (up to 16 digits)
            "pin": "",              # PIN (4 digits)
            "otherPII": [],         # List of non-financial PII elements
        }
        # Temporary storage for PII obtained in the current turn
        self.turn_pii = {key: "" for key in self.pii}

        # Utility function weights (Section 3.3)
        self.weights = {
            "wf": 1,    # Weight for financial PII component (I_f)
            "wo": 5,    # Weight for other PII component (I_o)
            "wt": 5,    # Weight for time penalty
        }

        # Termination payoffs (Equation 4)
        self.bonuses = {
            "hangup": 15,       # W_hangup: reward for rational disengagement
            "fullPII": 100,     # W_fullPII: reward for complete FPII extraction
            "endurance": 500,   # Reserved for extended interaction scenarios
        }

        # Saturation rate for non-financial PII (Equation 3)
        # k = 0.15 ensures 3-5 elements yield substantial value (I_o ~ 0.4-0.5)
        self.k = 0.15

    def calculate_f(self, turn, Tmax):
        """
        Calculate the financial PII extraction score (I_f).

        Evaluates each component of financial PII (name, expiry, card
        number, PIN) as a fraction of completeness. The score decreases
        with time to reflect mounting opportunity cost.

        Note: The full hyperbolic form from Equation 2 (1/(1-FPII)) is
        applied at the utility() level. This method computes the raw
        FPII component scores.

        Args:
            turn (int): Current turn index.
            Tmax (int): Maximum number of turns (default 50).

        Returns:
            float: Weighted financial PII score adjusted for time.
        """
        n = self.name_score()                        # Name completeness [0, 1]
        e = 1 if self.pii["expiryDate"] else 0       # Expiry: binary
        c = len(self.turn_pii["cardNumber"]) / 16     # Card number: fraction of 16 digits
        p = self.pin_score()                          # PIN completeness [0, 1]

        # Average across four FPII components, discounted by time elapsed
        F = (n + e + c + p) / 4 * (1 - turn / Tmax)
        return F

    def name_score(self):
        """
        Calculate name completeness score.

        Returns:
            float: 1.0 if a name has been disclosed, 0.0 otherwise.
        """
        return 1.0 if self.pii["nameOnBankCard"] else 0.0

    def pin_score(self):
        """
        Calculate PIN completeness score.

        Returns:
            float: Fraction of 4-digit PIN disclosed (0.0 to 1.0).
        """
        return len(self.pii["pin"]) / 4 if self.pii["pin"] else 0.0

    def calculate_o(self):
        """
        Calculate the non-financial PII score (I_o) using Equation 3.

        Uses exponential decay: I_o = 1 - exp(-k * r)
        where r is the count of unique non-financial PII elements.
        This encodes diminishing marginal returns for additional PII.

        Returns:
            float: Non-financial PII score in [0, 1).
        """
        r = len(set(self.turn_pii["otherPII"]))
        O = 1 - math.exp(-self.k * r)
        return O

    def utility(self, data_extracted, turn, hang_up=False, Tmax=50):
        """
        Compute the total scammer utility for a single turn (Equation 1).

        U_S(t) = w_f * I_f + w_o * I_o - w_t * (t / t_max) + W_s(t)

        The baiter's utility is the negative: U_B(t) = -U_S(t).

        Args:
            data_extracted (dict): PII extracted this turn, with keys
                matching the response_analysis_scheme.
            turn (int): Current turn index.
            hang_up (bool): Whether the scammer selected S10 this turn.
            Tmax (int): Maximum number of turns. Default: 50.

        Returns:
            float: The scammer's utility score for this turn.
        """
        # Update cumulative PII state with newly extracted data
        self.update_pii(data_extracted)

        # Compute utility components
        F = self.calculate_f(turn, Tmax)          # Financial PII score
        O = self.calculate_o()                    # Non-financial PII score
        t_penalty = self.weights["wt"] * (turn / Tmax)  # Time penalty

        # Weighted sum of components
        score = self.weights["wf"] * F + self.weights["wo"] * O - t_penalty

        # Add termination bonuses (Equation 4)
        if hang_up:
            score += self.bonuses["hangup"]
        if self.check_full_pii():
            score += self.bonuses["fullPII"]

        return score

    def update_pii(self, new_pii):
        """
        Update cumulative PII state with newly extracted data.

        Handles deduplication for non-financial PII and updates
        financial PII fields only when new values are received.

        Args:
            new_pii (dict): Newly extracted PII from this turn.
        """
        for key, value in new_pii.items():
            if key in self.pii and value:
                if key == "otherPII":
                    # Append only non-duplicate other PII entries
                    current_value = self.pii[key]
                    if value not in current_value:
                        self.pii[key].append(value)
                elif value != self.pii[key]:
                    # Update financial PII if the new value differs
                    self.pii[key] = value

    def check_full_pii(self):
        """
        Check whether complete financial PII (FPII) has been obtained.

        Complete FPII requires all four elements:
        - Cardholder name (non-empty)
        - Expiry date (non-empty)
        - Card number (exactly 16 digits)
        - PIN (exactly 4 digits)

        Returns:
            bool: True if all four FPII elements are complete.
        """
        return (
            bool(self.pii["nameOnBankCard"])
            and bool(self.pii["expiryDate"])
            and len(self.pii["cardNumber"]) == 16
            and len(self.pii["pin"]) == 4
        )


if __name__ == "__main__":
    # Interactive mode for testing utility computation
    calculator = UtilityCalculator()
    while True:
        data = eval(input("data: "))
        print("Utility:", calculator.utility(data, 1))
        print("Full PII obtained:", calculator.check_full_pii())
