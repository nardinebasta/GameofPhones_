# Bot Wars: Game-Theoretic Framework for LLM-Based Scam Baiting

**Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars' to Counteract Phone Scams**

This repository contains the source code for the Bot Wars framework, a game-theoretic system that enables Large Language Models (LLMs) to learn effective counter-strategies for automated scam baiting through in-context experience, without fine-tuning.

## Overview

Phone scams impose substantial costs on individuals and financial systems worldwide. Bot Wars models scammer-baiter interactions as a two-player non-cooperative zero-sum game, where:

- The **scammer** seeks to extract complete financial personally identifiable information (FPII)
- The **baiter** impersonates a potential victim to maximise interaction duration while preventing complete FPII disclosure

The framework enables LLMs to discover effective baiting strategies through repeated gameplay, with Nash equilibrium serving as a formal benchmark for evaluating learned behaviour.

### Key Results

- **300 simulated dialogues** across three LLM architectures (DeepSeek, Mixtral, GPT-4)
- Adaptive strategy selection achieves **28.9–79.7% closer alignment** to Nash equilibrium than non-adaptive baselines
- Reduces financial credential exposure by **up to 36%** compared to static prompting
- **10,247 annotated turns** with turn-level strategy labels and utility scores

## Architecture

Each simulated turn comprises five sequentially executed LLM calls:

```
┌─────────────────────┐     ┌─────────────────────┐
│  Scammer Strategy    │     │  Baiter Strategy     │
│  Selector            │     │  Selector            │
│  (ActionBot)         │     │  (ActionBot)         │
└────────┬────────────┘     └────────┬────────────┘
         │ S_i                       │ B_j
         ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Scammer Message     │     │  Baiter Message      │
│  Generator           │     │  Generator           │
│  (ResponseBot)       │     │  (ResponseBot)       │
└────────┬────────────┘     └────────┬────────────┘
         │ msg_S                     │ msg_B
         ▼                           ▼
       ┌───────────────────────────────┐
       │     Dialogue Analyser         │
       │  (PII Extraction & Utility)   │
       └───────────────────────────────┘
```

Strategy selectors process historical performance data from the 20 most recent games to output probability distributions over 10 strategy options. Message generators translate selected strategies into contextually appropriate dialogue. The dialogue analyser extracts PII and computes utility scores.

## Repository Structure

```
├── BotWarsGPT.py                 # Game orchestrator for GPT-4
├── BotWarsDeepSeek.py            # Game orchestrator for DeepSeek
├── BotWarsMixtral.py             # Game orchestrator for Mixtral
├── ActionBotGPT.py               # Strategy selector for GPT-4
├── ActionBotDeepSeek.py          # Strategy selector for DeepSeek
├── ActionBotMixtral.py           # Strategy selector for Mixtral
├── ResponseBotGPT.py             # Message generator for GPT-4
├── ResponseBotDeepSeek.py        # Message generator for DeepSeek
├── ResponseBotMixtral.py         # Message generator for Mixtral
├── ResponseDataAnalyser.py       # Dialogue analyser (PII extraction)
├── UtilityCalculator.py          # Turn-level utility computation (Eq. 1)
├── SystemPromptGenerator.py      # Prompt definitions for all components
├── PayoffMatrixGeneratorFinal.py # Empirical payoff matrix construction (Eq. 5)
├── NashCalculationFinal.py       # Nash equilibrium computation (Eqs. 6-7)
├── requirements.txt              # Python dependencies
├── .env.example                  # API key configuration template
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.9+
- API keys for the LLM providers you wish to use

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/bot-wars.git
   cd bot-wars
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys by copying the template and filling in your keys:
   ```bash
   cp .env.example .env
   ```

   Required keys depend on which architectures you want to run:
   - `OPENAI_API_KEY` — for GPT-4 experiments and the dialogue analyser
   - `DEEPSEEK_API_KEY` — for DeepSeek experiments
   - `OCTOAI_API_KEY` — for Mixtral experiments

4. Create data directories:
   ```bash
   mkdir -p data/gpt data/deepseek data/mixtral
   ```

### Running Experiments

To run 100 game simulations with a specific architecture:

```bash
# GPT-4 experiments
python BotWarsGPT.py

# DeepSeek experiments
python BotWarsDeepSeek.py

# Mixtral experiments
python BotWarsMixtral.py
```

Each run produces JSON game logs in the corresponding `data/` subdirectory.

### Analysis Pipeline

After running simulations, generate payoff matrices and compute Nash equilibrium:

```bash
# 1. Generate per-game payoff matrices from game logs
python PayoffMatrixGeneratorFinal.py

# 2. Compute Nash equilibrium from the average payoff matrix
python NashCalculationFinal.py
```

## Module Descriptions

| Module | Role | Description |
|--------|------|-------------|
| `BotWars*.py` | Orchestrator | Coordinates the 5-component turn pipeline, manages game state, handles logging and termination conditions |
| `ActionBot*.py` | Strategy Selector | Analyses interaction history and payoff feedback to select strategies from 10 options via in-context learning |
| `ResponseBot*.py` | Message Generator | Translates selected strategies into natural language dialogue using a two-layer prompt architecture |
| `ResponseDataAnalyser.py` | Dialogue Analyser | Extracts financial and non-financial PII from baiter responses using structured JSON output |
| `UtilityCalculator.py` | Utility Function | Computes the scammer's turn-level utility incorporating financial PII extraction, non-financial PII, time penalty, and termination bonuses |
| `SystemPromptGenerator.py` | Prompt Store | Defines all role-specific prompts for strategy selectors, message generators, and the dialogue analyser |
| `PayoffMatrixGeneratorFinal.py` | Matrix Builder | Constructs 10×10 empirical payoff matrices from game logs and computes cross-game averages |
| `NashCalculationFinal.py` | Equilibrium Solver | Computes Nash equilibrium mixed strategies via linear programming |

## Utility Function

The scammer's utility at turn *t* is:

```
U_S(t) = I_f + w_o · I_o − w_t · (t / t_max) + W_s(t)
```

where:
- **I_f** = Financial PII extraction score (hyperbolic in FPII completeness)
- **I_o** = Non-financial PII score with diminishing returns: `1 − exp(−k · r)`
- **w_t · (t/t_max)** = Linear time penalty
- **W_s(t)** = Termination bonus (+15 for hang-up, +100 for complete FPII)

Parameters: `t_max=50, w_o=5, w_t=5, k=0.15`

## Strategies

### Scammer Strategies (S1–S10)

| ID | Strategy | Description |
|----|----------|-------------|
| S1 | Urgency | Artificial time constraints |
| S2 | Authority | False institutional power |
| S3 | Emotional Manipulation | Evoke emotional responses |
| S4 | Incentive | Present rewards |
| S5 | Information Extraction | Acquire PII systematically |
| S6 | Technical Jargon | Impair critical thinking |
| S7 | Reassurance | Fabricated guarantees |
| S8 | Building Rapport | Cultivate connection |
| S9 | Persistence | Continuous pressure |
| S10 | Hang Up | Terminate conversation |

### Baiter Strategies (B1–B10)

| ID | Strategy | Description |
|----|----------|-------------|
| B1 | Delay | Prolong interaction |
| B2 | Obfuscation | Unclear responses |
| B3 | Technical Difficulty | Artificial obstacles |
| B4 | Counter Questioning | Challenge narrative |
| B5 | Information Fabrication | Provide false data |
| B6 | Malicious Compliance | Misconstrue instructions |
| B7 | Conversation Diversion | Disrupt scripts |
| B8 | Pretended Naivety | Feign misunderstanding |
| B9 | Verification Request | Challenge legitimacy |
| B10 | Reverse Engineering | Extract scammer info |

## Ethical Considerations

- All experiments use LLM-simulated interactions conducted under IRB approval
- No real scammers or victims were engaged
- The released dataset includes baiter strategies and performance metrics but excludes scammer prompts and adversarial content
- All LLM APIs are used in compliance with their respective terms of service

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{botwars2026,
  title={Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars' to Counteract Phone Scams},
  author={Anonymous},
  booktitle={ECML-PKDD 2026},
  year={2026}
}
```

## License

This project is released for academic research purposes. See the paper for full details on the dataset release protocol.
