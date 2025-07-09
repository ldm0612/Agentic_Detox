# COLX 565 Final Project: Sentiment, Toxicity Analysis, and Detoxification Pipeline

## Overview

This project provides a modular pipeline for analyzing the sentiment and toxicity (offensive, explicit language) of text data, with advanced support for detoxification (rewriting toxic text) and multilingual processing (English and Swahili). It leverages relatively light-weight, open-source large language models (LLMs) and modern NLP frameworks (LangChain, HuggingFace) to perform End-to-End sentiment analysis, toxicity analysis, and text detoxification (rewriting offensive and explicit language into a neutrual, friendly tone).

The project is organized into two main milestones, each conatains multiple tasks:
- **Milestone-1:**
  - Task 1: Sentiment Analysis
  - Task 2: Toxicity Analysis
- **Milestone-2:
  - Task 3: English Detoxification
  - Task 4: English/Swahili Identification
  - Task 5: Swahili Machine Translation
  - Task 6: End-to-End Workflow
    - Rule-based
    - Agentic

---

## Workflow

1. **Input:** Sentence(s) in English or Swahili.
2. **Language Identification:** Detect if the sentence is English or Swahili.
3. **Translation:** If Swahili, translate to English.
4. **Sentiment Analysis:** Classify as positive, negative, mixed, or neutral, with explanation.
5. **Toxicity Detection:** Classify as toxic or non-toxic, with explanation.
6. **Detoxification:** If toxic, rewrite to be non-offensive while preserving meaning.
7. **Output:** Structured result with all relevant fields and explanations.

The pipeline can be run in two modes:
- **Rule-based:** Deterministic sequence of utility functions. 
- **Agentic:** An LLM agent dynamically decides which tools to invoke at each step.

### Note

This project intentionally uses lightweight, open-source LLMs to ensure accessibility and efficient deployment within limited computational resources and budget. Future iterations could explore integrating larger models or API-based solutions to further enhance performance.

---

## Milestone-1: English Sentiment & Toxicity Analysis

**Goal:**  
Analyze the sentiment and toxicity of English sentences using an LLM.

**Features:**
- Sentiment analysis (positive, negative, mixed, neutral) with explanations.
- Toxicity detection (toxic, non-toxic) with explanations.
- Batch processing of CSV files.
- Explainable, structured outputs.

**Key Files:**
- `Milestone-1/Milestone-1.ipynb`: Notebook for developing the workflow, with title captions and example results.
- `Milestone-1/main.py`: CLI entry point.
- `Milestone-1/pipeline.py`: Orchestrates the workflow.
- `Milestone-1/sentiment.py`, `Milestone-1/toxicity.py`: Task logic.
- `Milestone-1/utils.py`: Data loading/utilities.
- `Milestone-1/prompt.py`: Prompt templates.

---

## Milestone-2: Multilingual, Detoxification, and Agentic Pipeline

**Goal:**  
Extend the pipeline for multilingual input (English/Swahili), detoxification, and agentic orchestration.

**Features:**
- **Language Identification:** Uses `xlm-roberta-base-language-detection` for English/Swahili detection.
- **Translation:** Swahili sentences are translated to English.
- **Detoxification:** Toxic sentences are rewritten to be non-offensive.
- **Rule-based Pipeline:** Deterministic workflow.
- **Agentic Pipeline:** LLM agent (via LangChain) orchestrates the workflow.
- **Comprehensive Output:** Language, sentiment, toxicity, detoxified text, and explanations.

**Key Files:**
- `Milestone-2/Milestone-1.ipynb`: Notebook for developing the workflow, with title captions and example results.
- `Milestone-2/rule_based_pipeline.py`: Rule-based workflow.
- `Milestone-2/agentic_pipeline.py`: Agentic workflow.
- `Milestone-2/language_id.py`: Language detection.
- `Milestone-2/translation.py`: Swahili-to-English translation.
- `Milestone-2/sentiment.py`, `Milestone-2/toxicity.py`, `Milestone-2/detox.py`: Task logic.
- `Milestone-2/prompt.py`: Prompt templates and agent instructions.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Agentic_Detox
   ```

2. **Install dependencies:**
   ```bash
   pip install -r Milestone-1/requirements.txt
   pip install -r Milestone-2/requirements.txt
   ```
   > **Note:** There is a typo in the root requirements file: use `requirements.txt` instead of `requriement.txt`.

3. **Prepare input data:**
   - Place your CSV files in the appropriate `data/` directories.

---

## Usage

### Milestone-1 (English only)

```bash
cd Milestone-1
python main.py --input data/milestone-1-eng_sentiment.csv --output data/milestone1_result/sentiment_analysis_results.csv
```

### Milestone-2 (Multilingual, Detoxification, Agentic)

- **Rule-based pipeline:**
  ```python
  from rule_based_pipeline import process_texts_rule_based
  results = process_texts_rule_based(list_of_sentences)
  ```

- **Agentic pipeline:**
  ```python
  from agentic_pipeline import process_texts_agentic
  results = process_texts_agentic(list_of_sentences)
  ```

---

## Data Format

- **Input:** CSV files with a `sentence` column.
- **Output:** CSV or Python dictionaries with fields:
  - `original_sentence`
  - `translated_sentence` (if applicable)
  - `sentiment_prediction` and `sentiment_prediction_explanation`
  - `toxicity_prediction` and `toxicity_prediction_explanation`
  - `detoxed_sentence` (if applicable)

---

## Models Used

- **Sentiment & Toxicity:** IBM Granite-3.2-2B-Instruct (https://huggingface.co/ibm-granite/granite-3.2-2b-instruct)
- **Language Identification:** xlm-roberta-base-language-detection (https://huggingface.co/papluca/xlm-roberta-base-language-detection)
- **Translation:** Swahili-to-English translation model (https://huggingface.co/UBC-NLP/toucan-base)

---

## Extensibility

- Modular design: add new languages, tasks, or models easily.
- Customizable prompts and output schemas.
- Supports both deterministic and agentic orchestration.


*For more details, see the code and Jupyter notebooks in each milestone directory.*
