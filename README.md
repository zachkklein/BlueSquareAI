# BlueSqureAI

An AI-powered system that analyzes text to detect antisemitic rhetoric using RAG (Retrieval-Augmented Generation) and multi-step reasoning.

Entry in the Marshall Wace field of Hack@Brown 2025

---

## Features

- **Comprehensive Trope Detection**: Identifies 10 types of antisemitic tropes including:
  - Elite control conspiracies
  - Dual loyalty accusations
  - Collective guilt
  - Financial conspiracies
  - Blood libel
  - Holocaust denial/distortion
  - Proxy figures
  - Dog whistles
  - Religious demonization

- **Multi-Factor Scoring**: Risk assessment considers:
  - Trope strength and type
  - Counterfactual reasoning
  - Target explicitness
  - Language explicitness

---

## Demo

**Risk Score:** 0.70 / 1.0  
**Verdict:** High-risk trope-based rhetoric  
**Detected Trope:** `elite_control`  
**Trope Strength:** 0.70  

**Explanation:**  
The claim could be interpreted as a critique of the influence of financial institutions on media without implying an ethnic conspiracy.  

**Reasoning:**  
The claim:

> "The group of bankers control the media"

closely resembles the **'elite control' trope**, which suggests that Jews collectively control powerful institutions like media and finance. The use of terms like *"bankers"* and *"control the media"* aligns with historical antisemitic narratives. However, without explicit mention of Jews, it could also be interpreted as a general critique of financial influence on media, making the resemblance strong but not definitive.

---

# Detailed Analysis

- **Extracted Claim:** The group of bankers control the media  
- **Target:** `implicit_jews`  
- **Explicitness:** Implicit  
- **Counterfactual:** A group controls the media  
- **Meaning Preserved:** False  

**Counterfactual Explanation:**  
The original claim specifically identifies *"bankers"* as the group in control, which may carry implicit identity-based assumptions or stereotypes. By replacing this specific identity with a neutral term, the claim loses its specific connotation and potential implications about the group's identity, thus changing the meaning.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zachkklein/BlueSquareAI
cd blueSquareAI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

---


## Quick Start

### Basic Usage

```python
from pipeline.aggregate import classify_text

result = classify_text("They control the media narrative.")

print(f"Risk Score: {result['risk_score']}")
print(f"Verdict: {result['verdict']}")
print(f"Trope: {result['trope']}")
```

### Async Usage

```python
from pipeline.aggregate import classify_text_async
import asyncio

async def main():
    result = await classify_text_async("They control the media narrative.")
    print(result)

asyncio.run(main())
```

### Batch Processing

```python
from pipeline.aggregate import classify_texts_batch
import asyncio

texts = [
    "They control the media.",
    "The Rothschilds control banking.",
    "All Jews are responsible for Israel's actions."
]

results = asyncio.run(classify_texts_batch(texts))
for text, result in zip(texts, results):
    print(f"{text}: {result['risk_score']:.2f}")
```

## Output Format

```python
{
    "verdict": "High-risk trope-based rhetoric",
    "risk_score": 0.75,
    "trope": "elite_control",
    "trope_strength": 0.8,
    "explanation": "Alternative interpretation...",
    "reasoning": "Brief explanation...",
    "details": {
        "extracted_claim": "...",
        "target": "implicit_jews",
        "explicitness": "implicit",
        "counterfactual": "...",
        "meaning_preserved": False,
        "counterfactual_explanation": "..."
    }
}
```

---


## Evaluation

Run the evaluation script to assess system performance:

```bash
python evaluate.py
```

This will:
- Test the system on evaluation data
- Calculate comprehensive metrics (MAE, RMSE, R², Accuracy, Precision, Recall, F1)
- Generate visualizations
- Save detailed results to `evaluation_results.json`

---


## Architecture

The system uses a multi-stage pipeline:

1. **Claim Extraction**: Extracts main claims and identifies targets
2. **Context Retrieval**: Retrieves relevant knowledge base documents using RAG
3. **Trope Mapping**: Maps claims to known antisemitic tropes
4. **Counterfactual Testing**: Tests if claims depend on identity-based meaning
5. **Risk Scoring**: Multi-factor risk score calculation

---


### Pipeline Components

- `pipeline/aggregate.py` - Main entry point
- `pipeline/extract_claim.py` - Claim extraction
- `pipeline/retrieve_context.py` - RAG-based context retrieval
- `pipeline/map_trope.py` - Trope identification
- `pipeline/counterfactual.py` - Counterfactual reasoning
- `pipeline/aggregate_optimized.py` - Optimized async implementation

---


### Knowledge Base

The `kb/` directory contains reference materials on:
- IHRA definition of antisemitism
- Various antisemitic tropes
- Guidelines for distinguishing criticism from antisemitism

---


## Requirements

- Python 3.8+
- OpenAI API key (via OpenRouter)
- See `requirements.txt` for full dependencies

---


## Project Structure

```
blueSquareAI/
├── pipeline/              # Core pipeline modules
│   ├── aggregate.py       # Main entry point
│   ├── aggregate_optimized.py  # Optimized async implementation
│   ├── extract_claim.py   # Claim extraction
│   ├── extract_claim_async.py
│   ├── retrieve_context.py  # RAG-based retrieval
│   ├── map_trope.py       # Trope identification
│   ├── map_trope_async.py
│   ├── counterfactual.py  # Counterfactual reasoning
│   └── counterfactual_async.py
├── kb/                       # Knowledge base (trope definitions)
├── eval_data.py             # Evaluation dataset
├── evaluate.py              # Evaluation script
├── liveDemo.ipynb           # Demo for judging presentation
└── README.md
```

---


## License

Apache 2.0

---


## Acknowledgments

Built for the Marshall Wace track at Hack@Brown Hackathon. Uses the IHRA definition of antisemitism as a reference framework.

Thanks to Sachin and Henry from Marshall Wace for their workshop on AI which helped me understand pipelines and RAG

Used ChatGPT, Google Gemini, and Cursor for help in building, documenting, and testing the code
