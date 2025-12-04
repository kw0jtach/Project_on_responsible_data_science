# Task 6: Explainability and LLMs

## Overview

This document describes our approach to building a **natural language interface** for presenting machine learning explanations in human-friendly text, as required by Task 6 of the Responsible Data Science project.

---

## Objective

> "Choose an explainability method. Build a natural language interface to present explanations (e.g., feature importances or example-based explanations) in human-friendly text."

Our goal is to transform the technical outputs from Task 5 (Explainability) into accessible, conversational explanations that non-technical users can understand.

---

## Design Decisions

### 1. Chosen Explainability Method: **LIME**

We selected **LIME (Local Interpretable Model-agnostic Explanations)** as our primary explainability method for the LLM interface.

#### Why LIME?

| Criterion | LIME | SHAP | Counterfactuals |
|-----------|------|------|-----------------|
| **Instance-specific** | ✅ Explains individual predictions | ⚠️ Better for global patterns | ✅ Instance-specific |
| **Human-relatable** | ✅ "Feature X pushed toward class Y" | ⚠️ More abstract (Shapley values) | ⚠️ Can be confusing |
| **Actionable output** | ✅ Clear feature contributions | ⚠️ Harder to translate | ✅ "Change X to Y" |
| **LLM-friendly format** | ✅ Simple feature-importance pairs | ⚠️ Complex nested structure | ⚠️ Requires interpretation |

**LIME provides:**
- Per-feature importance scores with direction (positive/negative)
- Direct mapping to natural language ("high capital-gain strongly indicates high income")
- Instance-level granularity perfect for personalized explanations

### 2. Architecture: **Direct Context Injection (No RAG)**

We chose **not** to implement a Retrieval-Augmented Generation (RAG) system.

#### Why Not RAG?

| RAG is Ideal When... | Our Situation |
|---------------------|---------------|
| Large corpus of documents | ~10 explained instances + 1 summary table |
| Semantic search needed | We know exactly which data to retrieve |
| Questions span many topics | Questions are about specific predictions |
| Dynamic knowledge base | Static, pre-computed explanations |

**Our data footprint:**
- 10 LIME explanations (one per confident mistake)
- 1 SHAP global importance table (30 features)
- 1 Counterfactual summary
- 1 Privacy comparison table

This entire dataset fits comfortably within a single LLM prompt context window (~4K tokens). Adding RAG would introduce:
- Unnecessary complexity (embeddings, vector stores, retrieval logic)
- Potential retrieval errors
- Higher latency
- No practical benefit

#### Our Approach: Context-Injected Prompts

```
┌─────────────────────────────────────────┐
│           User Query                    │
│  "Why was instance 29306 predicted      │
│   as high income?"                      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Python Extraction Layer            │
│  - Retrieve LIME data for instance      │
│  - Format feature importances           │
│  - Get instance metadata                │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Prompt Construction             │
│  System: "You are an ML explainer..."   │
│  Context: [LIME features + values]      │
│  User: [Original question]              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       LLM (Gemma via LM Studio)         │
│  Generates human-friendly explanation   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Natural Language Output         │
│  "This person was predicted to earn     │
│   >$50K primarily because of their      │
│   high capital gains ($34,095)..."      │
└─────────────────────────────────────────┘
```

### 3. LLM Choice: **Google Gemma (via LM Studio)**

We use a locally-hosted LLM through LM Studio:
- **Model**: `google/gemma-3-12b` (or similar Gemma variant)
- **Interface**: OpenAI-compatible API at `http://127.0.0.1:1234/v1`
- **Advantages**:
  - No API costs or rate limits
  - Data stays local (privacy-friendly for our sensitive dataset)
  - Consistent availability during development

---

## Implementation Structure

### Two-Tier Approach

#### Tier 1: Simple Function Interface (Baseline)

```python
def explain_instance(instance_id: int) -> str:
    """Generate a human-friendly explanation for a specific prediction."""
    # 1. Extract LIME data for the instance
    # 2. Build context prompt
    # 3. Call LLM
    # 4. Return natural language explanation
```

**Use case**: Quick, programmatic access to explanations.

#### Tier 2: Interactive Chat Interface (Advanced)

```python
def chat_explain(question: str, messages: list) -> tuple[str, list]:
    """Multi-turn conversation about model explanations."""
    # Maintains conversation history
    # Supports follow-up questions
    # Can switch between explanation modes
```

**Use case**: Exploratory analysis, follow-up questions, deeper understanding.

---

## Prompt Engineering

### System Prompt

The system prompt establishes the LLM's role and constraints:

```
You are an AI assistant that explains machine learning predictions in simple, 
human-friendly language. You have access to LIME explanations that show which 
features influenced a prediction and by how much.

Guidelines:
- Use plain language, avoid jargon
- Explain feature contributions in terms of real-world meaning
- Be honest about uncertainty
- Focus on the most important factors (top 3-5 features)
- Provide actionable insights when relevant
```

### Context Template

For each instance explanation, we inject:

```
PREDICTION DETAILS:
- Instance ID: {id}
- Predicted Class: {prediction} (Confidence: {confidence}%)
- True Label: {true_label}
- This prediction was: {CORRECT/INCORRECT}

LIME FEATURE CONTRIBUTIONS (most important first):
1. {feature_1}: {importance_1} ({direction})
2. {feature_2}: {importance_2} ({direction})
...

INSTANCE ATTRIBUTES:
- Age: {age_binary} ({"older" if 1 else "younger"} than 38)
- Sex: {sex} ({"Male" if 1 else "Female"})
- Education: {education}
- Occupation: {occupation}
...
```

---

## Data Flow

### From Explainability.ipynb to LLM Interface

1. **Task 5** generates:
   - `top_10_mistakes` DataFrame (confident mistakes)
   - `local_explanations["lime"]` (LIME explanations per instance)
   - `importance_df_complete` (SHAP global importance)
   - `comparison_df` (privacy impact analysis)

2. **Task 6** consumes this data by:
   - Re-running data preparation steps (for fresh instances)
   - Or loading pre-computed explanations from memory/disk

3. **LLM Interface** transforms:
   - Technical LIME output → Natural language explanation
   - Feature importance scores → "Feature X strongly suggests Y"
   - Prediction probabilities → "The model was 99% confident"

---

## Example Interactions

### Query 1: Instance Explanation

**User**: "Why was instance 29306 predicted as high income?"

**LLM Response**:
> "This person was predicted to earn more than $50,000 primarily because of their exceptionally high capital gains of $34,095. This single factor was so influential that it overshadowed all other characteristics. The model associates large investment returns with high-income individuals, which makes sense - capital gains typically require significant wealth to accumulate.
>
> However, this prediction was actually **incorrect**. The person's true income was below $50K, suggesting that this capital gain might have been a one-time event (like selling a house) rather than regular income."

### Query 2: Global Model Behavior

**User**: "What features does the model care about most?"

**LLM Response**:
> "The model primarily focuses on economic and social factors rather than demographics:
>
> 1. **Marital Status** (most important): Being married to a civilian spouse is the strongest predictor of high income, likely reflecting dual-income households.
>
> 2. **Education Level**: More years of education strongly correlates with higher income predictions.
>
> 3. **Capital Gains**: Investment income signals existing wealth.
>
> Interestingly, age and sex - the attributes we applied privacy protection to - rank only #14 and #29 respectively. This means the model has learned meaningful economic patterns rather than relying on demographic shortcuts."

### Query 3: Counterfactual Advice

**User**: "What would person 9839 need to change to get a different prediction?"

**LLM Response**:
> "According to the counterfactual analysis, the single most impactful change would be their work status. Currently, this person is marked as 'Without-pay' employment, which strongly signals low income to the model.
>
> If they were to move to paid employment (changing workclass from 'Without-pay' to any paid category), the prediction would flip from low income to high income. No other changes would be necessary.
>
> This suggests the model views unpaid work as a definitive indicator of low income, regardless of other factors like education or family situation."

---

## Limitations and Considerations

### 1. Explanation Fidelity
LIME explanations are **approximations** of the model's behavior around a specific instance. They may not perfectly capture the true decision boundary.

### 2. LLM Hallucination Risk
The LLM might embellish or misinterpret the provided context. We mitigate this by:
- Providing structured, factual context
- Constraining the system prompt
- Keeping explanations grounded in the data

### 3. Static Explanations
Our current implementation uses pre-computed explanations. Real-time LIME computation would require additional setup but would support any instance.

### 4. Model Dependency
The explanations are specific to **THE PRIVATE CLASSIFIER**. Different models would produce different LIME explanations.

---

## Conclusion

Our approach prioritizes **simplicity, reliability, and user-friendliness**:

- **LIME** provides the most human-relatable explanation format
- **Direct context injection** avoids RAG complexity while fully utilizing our small dataset
- **Local LLM** (Gemma via LM Studio) ensures privacy and availability
- **Two-tier interface** (simple + interactive) accommodates different use cases

This design transforms technical ML outputs into accessible natural language, making model behavior understandable to non-technical stakeholders - a key goal of responsible AI deployment.
