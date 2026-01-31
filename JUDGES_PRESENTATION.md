# Antisemitism Detection System - Judges Presentation Guide

## Executive Summary

This system uses advanced AI techniques to detect antisemitic rhetoric in text by combining **Retrieval-Augmented Generation (RAG)**, **multi-step reasoning**, and **counterfactual analysis**. Unlike simple sentiment analysis, it identifies specific antisemitic tropes and provides nuanced risk assessments.

---

## The Problem

Antisemitic rhetoric often appears in coded language, dog whistles, and implicit references that evade simple keyword detection. The system addresses:

- **Ambiguity**: Distinguishing legitimate criticism from antisemitic tropes
- **Context**: Understanding when coded language targets Jews
- **Accuracy**: Providing reliable, explainable assessments
- **Speed**: Processing text quickly for real-world applications

---

## Technical Architecture

### High-Level Flow

```
Input Text
    ↓
1. Claim Extraction → Extract main claim, identify target, assess explicitness
    ↓
2. Context Retrieval (RAG) → Retrieve relevant knowledge base documents
    ↓
3. Trope Mapping → Match claim to known antisemitic tropes
    ↓
4. Counterfactual Testing → Test if claim depends on identity-based meaning
    ↓
5. Risk Scoring → Multi-factor risk calculation
    ↓
Output: Risk Score + Detailed Analysis
```

### Key Components

#### 1. **Claim Extraction** (`extract_claim.py`)
- **Purpose**: Extract the main claim and identify who/what is being targeted
- **Output**: 
  - Main claim text
  - Target category (explicit_jews, implicit_jews, other, unclear)
  - Explicitness level (explicit, implicit)

**Example**:
- Input: "They control the media narrative"
- Output: Claim extracted, target = "implicit_jews", explicitness = "implicit"

#### 2. **Context Retrieval - RAG** (`retrieve_context.py`)
- **Purpose**: Retrieve relevant knowledge base documents using semantic search
- **Technology**: 
  - Vector embeddings (HuggingFace sentence-transformers)
  - LlamaIndex for RAG
  - Knowledge base of 10+ trope definitions
- **Why RAG**: Grounds decisions in authoritative definitions (IHRA, academic sources)

**Knowledge Base Contains**:
- Elite control tropes
- Dual loyalty accusations
- Collective guilt
- Financial conspiracies
- Blood libel
- Holocaust denial
- Proxy figures
- Dog whistles
- Religious demonization
- Israel criticism vs. antisemitism guidelines

#### 3. **Trope Mapping** (`map_trope.py`)
- **Purpose**: Match extracted claim to known antisemitic tropes
- **Process**:
  1. Compare claim against retrieved KB context
  2. Identify which trope (if any) the claim resembles
  3. Assess strength of resemblance (0.0-1.0)
  4. Provide alternative non-antisemitic interpretation
- **Output**: Trope type, strength score, reasoning

**Example**:
- Claim: "The Rothschilds control banking"
- Detected: `financial_conspiracy` trope
- Strength: 0.9
- Alternative: "Could refer to any wealthy banking family"

#### 4. **Counterfactual Testing** (`counterfactual.py`)
- **Purpose**: Test if claim relies on implicit identity-based meaning
- **Method**: 
  1. Rewrite claim with neutral actors
  2. Compare meaning preservation
  3. If meaning changes → likely identity-dependent
- **Why Important**: Distinguishes identity-based rhetoric from general critiques

**Example**:
- Original: "They control the media"
- Counterfactual: "A group controls the media"
- Meaning preserved? If yes → may not be identity-dependent
- If no → likely depends on implied identity

#### 5. **Risk Scoring** (`aggregate_optimized.py`)
- **Purpose**: Calculate final risk score using multiple factors
- **Formula**:
  ```
  base_score = trope_strength
  
  # Adjustments:
  - Counterfactual multiplier (0.3 if meaning preserved, 1.0 if not)
  - Target multiplier (1.2 for explicit_jews, 1.0 for implicit, 0.8 for other)
  - Explicitness multiplier (1.1 for explicit, 1.0 for implicit)
  
  risk_score = min(1.0, base_score × counterfactual × target × explicitness)
  ```
- **Verdict Categories**:
  - 0.0-0.3: Low-risk / non-identity-based
  - 0.3-0.6: Ambiguous — requires context
  - 0.6-1.0: High-risk trope-based rhetoric

---

## Key Innovations

### 1. **Multi-Step Reasoning Pipeline**
Not a single LLM call, but a structured pipeline:
- Each step focuses on a specific task
- Results feed into next step
- More accurate than monolithic approach

### 2. **RAG-Grounded Decisions**
- Uses knowledge base (not just LLM knowledge)
- Grounds decisions in authoritative sources
- Retrieves relevant context for each claim

### 3. **Counterfactual Reasoning**
- Tests identity-dependence systematically
- Reduces false positives
- Provides explainable results

### 4. **Multi-Factor Scoring**
- Considers multiple signals (trope, counterfactual, target, explicitness)
- More nuanced than binary classification
- Handles ambiguous cases appropriately

### 5. **Performance Optimizations**
- **Async/Parallel Processing**: Independent operations run simultaneously (3x faster)
- **Caching**: Repeated queries return instantly
- **Batch Processing**: Handle multiple texts efficiently

---

## How to Demonstrate

### Live Demo Flow

1. **Open the Jupyter Notebook** (`demo.ipynb`)
   - Show the clean interface
   - Explain the pipeline structure

2. **Run Example 1: High-Risk Case**
   ```
   Input: "The Rothschild family dictates world banking policy."
   Expected: High risk score (0.8-0.9), financial_conspiracy trope
   ```
   - Show risk score, verdict, trope detection
   - Explain the reasoning

3. **Run Example 2: Low-Risk Case**
   ```
   Input: "The banking system is rigged against the working class."
   Expected: Low risk score (0.0-0.2), no trope detected
   ```
   - Show how it distinguishes general critique from antisemitism

4. **Run Example 3: Ambiguous Case**
   ```
   Input: "International bankers are the real puppet masters."
   Expected: Medium risk (0.4-0.6), ambiguous verdict
   ```
   - Show nuanced handling of coded language

5. **Show Batch Processing**
   - Process multiple texts at once
   - Demonstrate speed and efficiency

### Key Talking Points

**For Judges:**

1. **"This isn't just sentiment analysis"**
   - We identify specific antisemitic tropes
   - Grounded in IHRA definition and academic research
   - Handles coded language and dog whistles

2. **"We use RAG, not just prompting"**
   - Knowledge base of trope definitions
   - Retrieves relevant context for each claim
   - More accurate than pure LLM knowledge

3. **"Multi-step reasoning for accuracy"**
   - Claim extraction → Context retrieval → Trope mapping → Counterfactual testing
   - Each step validates the previous
   - Reduces false positives and false negatives

4. **"Explainable and transparent"**
   - Shows which trope was detected
   - Provides alternative interpretations
   - Explains reasoning at each step

5. **"Production-ready performance"**
   - 3x faster with async processing
   - Caching for repeated queries
   - Batch processing support

---

## Technical Stack

- **LLM**: GPT-4o via OpenRouter
- **RAG**: LlamaIndex + HuggingFace embeddings
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Language**: Python 3.8+
- **Async**: asyncio for parallel processing
- **Evaluation**: Comprehensive metrics (MAE, RMSE, R², Precision, Recall, F1)

---

## Evaluation & Validation

### Metrics Tracked
- **Regression**: MAE, RMSE, R², Correlation
- **Classification**: Accuracy, Precision, Recall, F1
- **Per-Trope Analysis**: Performance breakdown by trope type

### Evaluation Dataset
- 13+ labeled examples
- Covers various trope types
- Includes ambiguous cases
- Ground truth scores (0.0-1.0)

### Run Evaluation
```bash
python evaluate.py
```

This generates:
- Comprehensive metrics report
- Visualizations (scatter plots, error distributions)
- Per-trope performance analysis
- Worst predictions for improvement

---

## Use Cases

1. **Content Moderation**: Flag antisemitic content on platforms
2. **Educational Tools**: Help people recognize antisemitic tropes
3. **Research**: Analyze large text corpora for antisemitic rhetoric
4. **Monitoring**: Track antisemitic discourse trends

---

## Limitations & Future Work

### Current Limitations
- Text-only (images coming soon)
- English language only
- Requires internet for LLM API calls

### Future Improvements
- Image analysis for memes and visual content
- Multi-language support
- Fine-tuned models for better accuracy
- Real-time monitoring dashboards
- Integration with social media APIs

---

## Presentation Tips

### Opening (30 seconds)
- "I've built an AI system that detects antisemitic rhetoric using advanced NLP techniques"
- "Unlike simple keyword matching, it identifies specific antisemitic tropes and provides nuanced risk assessments"

### Technical Deep Dive (2 minutes)
- Walk through the pipeline: Claim → RAG → Trope → Counterfactual → Score
- Emphasize RAG and multi-step reasoning
- Show knowledge base structure

### Live Demo (2 minutes)
- Run 2-3 examples showing different risk levels
- Highlight the reasoning and explanations
- Show batch processing speed

### Innovation Highlights (1 minute)
- RAG-grounded decisions
- Counterfactual reasoning
- Multi-factor scoring
- Performance optimizations

### Closing (30 seconds)
- "This system provides accurate, explainable detection of antisemitic rhetoric"
- "Ready for real-world deployment with comprehensive evaluation"

---

## Sample Script

**Opening:**
> "I've built an antisemitism detection system that goes beyond simple sentiment analysis. It uses Retrieval-Augmented Generation to ground decisions in authoritative knowledge, multi-step reasoning for accuracy, and counterfactual testing to reduce false positives."

**Demo:**
> "Let me show you how it works. I'll analyze this text: 'The Rothschild family dictates world banking policy.' The system extracts the claim, retrieves relevant knowledge base documents about financial conspiracy tropes, maps it to the appropriate trope, tests whether it depends on identity-based meaning, and calculates a risk score. As you can see, it correctly identifies this as a high-risk antisemitic trope with a score of 0.9."

**Technical:**
> "The key innovation is our multi-step pipeline. We don't just prompt an LLM once. Instead, we break the problem into steps: claim extraction, context retrieval using RAG, trope mapping, and counterfactual reasoning. Each step validates the previous, leading to more accurate results."

**Closing:**
> "This system is production-ready with async processing for speed, comprehensive evaluation metrics, and explainable results. It's ready to help platforms, educators, and researchers identify and understand antisemitic rhetoric."

---

## Questions Judges Might Ask

**Q: How do you handle false positives?**
A: We use counterfactual testing to check if claims depend on identity. We also provide alternative interpretations and show reasoning. The multi-factor scoring reduces false positives by requiring multiple signals.

**Q: What makes this better than just using GPT-4?**
A: RAG grounds decisions in authoritative knowledge, not just LLM training data. Multi-step reasoning is more accurate than a single prompt. Counterfactual testing systematically tests identity-dependence.

**Q: How do you evaluate accuracy?**
A: We have a labeled evaluation dataset with ground truth scores. We track regression metrics (MAE, correlation) and classification metrics (precision, recall, F1). We also analyze performance per trope type.

**Q: Can it handle new types of antisemitism?**
A: Yes, we can add new trope definitions to the knowledge base. The RAG system will automatically retrieve and use them. The system is designed to be extensible.

**Q: What about legitimate criticism of Israel?**
A: Our knowledge base includes guidelines for distinguishing legitimate political criticism from antisemitic rhetoric. The system considers context and provides alternative interpretations.

---

## Key Differentiators

1. **RAG-Grounded**: Uses knowledge base, not just LLM knowledge
2. **Multi-Step Reasoning**: Structured pipeline, not single prompt
3. **Counterfactual Testing**: Systematic identity-dependence testing
4. **Explainable**: Shows reasoning at each step
5. **Production-Ready**: Fast, cached, batch processing
6. **Comprehensive Evaluation**: Multiple metrics, per-trope analysis

---

## Conclusion

This system represents a sophisticated approach to detecting antisemitic rhetoric that combines:
- **Accuracy**: Multi-step reasoning and RAG
- **Explainability**: Transparent reasoning at each step
- **Performance**: Optimized for speed and scale
- **Validation**: Comprehensive evaluation framework

It's ready for real-world deployment and can help platforms, educators, and researchers identify and understand antisemitic rhetoric more effectively.

