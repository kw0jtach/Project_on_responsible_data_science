# Responsible Data Science Project

## Overview

This project explores responsible data science principles through a comprehensive machine learning pipeline using the Adult Income Dataset. The project combines classification, fairness assessment, privacy protection, explainability analysis, and natural language interfaces to demonstrate ethical AI practices.

**Goal**: Train classifiers to predict income (>50K) while studying the interplay between fairness, privacy, and explainability in machine learning systems.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Tasks Overview](#tasks-overview)
- [Usage Instructions](#usage-instructions)
- [Key Features](#key-features)
- [Technical Details](#technical-details)
- [Results Summary](#results-summary)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Project Structure

```
Project_on_responsible_data_science/
├── README.md                    # This file
├── project_statement.md         # Original project requirements
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── doc/                        # Documentation
│   ├── planning_overview.md    # Project planning and workflow
│   └── explainability-llms.md  # LLM interface design decisions
│
├── Classification.ipynb        # Task 1: Base classifier training
├── Fairness.ipynb             # Task 2: Fairness assessment and mitigation
├── DifferentialPrivacy.ipynb  # Task 3: Privacy protection implementation
├── Privacy_and_fairness.ipynb # Task 4: Combined privacy and fairness
├── Explainability.ipynb       # Task 5: Model explainability analysis
└── LLM-RAG.ipynb             # Task 6: Natural language explanations
```

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Project_on_responsible_data_science
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter lab
   ```

### Optional: LLM Setup for Task 6

For the natural language explanation interface:

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai)
2. **Download a Gemma model** (e.g., `google/gemma-3-12b`)
3. **Start the local server** in LM Studio (Developer tab → Start Server)
4. **Verify connection** at `http://127.0.0.1:1234`

## Tasks Overview

### Task 1: Classification
**File**: `Classification.ipynb`

- Preprocesses the Adult dataset with custom age binarization
- Trains a baseline logistic regression classifier
- Evaluates performance on test set
- Saves "THE CLASSIFIER" for reuse across tasks

**Key Outputs**:
- Base classifier with ~85% accuracy
- Standardized preprocessing pipeline
- Performance metrics and confusion matrices

### Task 2: Fairness
**File**: `Fairness.ipynb`

- Assesses group fairness using Age and Sex as protected attributes
- Implements Reweighing technique for bias mitigation
- Compares fairness metrics before and after mitigation
- Creates the "Fair Classifier"

**Key Metrics**:
- Statistical Parity Difference
- Disparate Impact
- Equal Opportunity Difference
- Average Odds Difference

### Task 3: Privacy
**File**: `DifferentialPrivacy.ipynb`

- Applies Local Differential Privacy to Age and Sex attributes
- Implements randomized response mechanism
- Analyzes privacy-utility trade-offs across multiple epsilon values
- Trains "Private Classifier" on noisy data

**Key Features**:
- Configurable privacy levels (ε = 0.1 to 5.0)
- Cross-tabulation analysis with error quantification
- Performance comparison with baseline classifier

### Task 4: Privacy and Fairness
**File**: `Privacy_and_fairness.ipynb`

- Combines differential privacy with fairness mitigation
- Creates "Private+Fair Classifier" using reweighing on private data
- Evaluates fairness using true (non-private) protected attributes
- Analyzes the interaction between privacy and fairness

**Key Insights**:
- Privacy-fairness trade-off analysis
- Auditor perspective on fairness measurement
- Comparative evaluation across all classifier variants

### Task 5: Explainability
**File**: `Explainability.ipynb`

- Studies explainability of the Private Classifier
- Identifies confident mistakes (high confidence, wrong predictions)
- Uses multiple explanation methods: SHAP, LIME, Counterfactuals
- Investigates impact of privacy noise on model errors

**Explanation Methods**:
- **Global**: SHAP feature importance
- **Local**: LIME instance explanations
- **Counterfactual**: What-if scenario analysis

### Task 6: LLM Interface
**File**: `LLM-RAG.ipynb`

- Builds natural language interface for ML explanations
- Uses LIME explanations with local LLM (Gemma via LM Studio)
- Provides both simple function and interactive chat interfaces
- Transforms technical outputs into human-friendly text

**Interface Types**:
- Simple function: `explain_instance(id)`
- Interactive chat: Multi-turn conversations about predictions

## Usage Instructions

### Sequential Execution (Recommended)

Run notebooks in order for full project workflow:

1. **Start with Classification**: Establishes baseline and saves artifacts
2. **Run Fairness**: Loads base classifier, adds fairness analysis
3. **Execute Privacy**: Creates private dataset variants
4. **Combine in Privacy_and_fairness**: Integrates privacy and fairness
5. **Analyze with Explainability**: Deep-dive into model behavior
6. **Interface with LLM-RAG**: Natural language explanations

### Independent Task Execution

Most notebooks can run independently if you only need specific analyses:

- **Fairness analysis only**: Run `Classification.ipynb` → `Fairness.ipynb`
- **Privacy analysis only**: Run `Classification.ipynb` → `DifferentialPrivacy.ipynb`
- **Explainability only**: Run `Classification.ipynb` → `Explainability.ipynb`

### Key Parameters to Modify

**Privacy Levels** (in `DifferentialPrivacy.ipynb`):
```python
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Adjust privacy levels
```

**Protected Groups** (in `Fairness.ipynb`):
```python
privileged_groups = [{'sex': 1, 'age_binary': 1}]    # Older males
unprivileged_groups = [{'sex': 0, 'age_binary': 0}]  # Younger females
```

**Explanation Methods** (in `Explainability.ipynb`):
```python
explainers = ['shap', 'lime', 'counterfactual']  # Choose methods
```

## Key Features

### Responsible AI Principles

- **Fairness**: Bias detection and mitigation using AIF360
- **Privacy**: Differential privacy with configurable protection levels
- **Explainability**: Multi-method approach (SHAP, LIME, Counterfactuals)
- **Transparency**: Natural language interfaces for non-technical users

### Technical Highlights

- **Consistent Preprocessing**: Custom pipeline ensures reproducibility
- **Modular Design**: Each task builds on previous outputs
- **Comprehensive Evaluation**: Performance, fairness, and privacy metrics
- **Interactive Explanations**: LLM-powered natural language interface

### Dataset Preprocessing

**Adult Income Dataset** with custom transformations:
- **Age**: Binarized at median (38 years) → `age_binary`
- **Sex**: Male=1, Female=0
- **Race**: White=1, Non-White=0
- **Target**: Income >50K = 1, ≤50K = 0

## Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**: Custom Adult dataset transformations
2. **Feature Scaling**: StandardScaler for numerical stability
3. **Model Training**: Logistic Regression with consistent parameters
4. **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1)

### Privacy Implementation

**Randomized Response Mechanism**:
- Combines Age and Sex into 4-category variable
- Applies noise based on epsilon parameter
- Maintains statistical utility while protecting individuals

### Fairness Metrics

- **Statistical Parity**: Equal positive prediction rates across groups
- **Disparate Impact**: Ratio of positive rates (legal threshold: 0.8)
- **Equal Opportunity**: Equal true positive rates for qualified individuals
- **Average Odds**: Equal TPR and FPR across groups

### Explainability Methods

- **SHAP**: Game-theoretic feature attributions
- **LIME**: Local linear approximations
- **Counterfactuals**: Minimal changes for different predictions

## Results Summary

### Model Performance

| Classifier | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|---------|
| Base | 0.8547 | 0.7234 | 0.6189 | 0.6667 |
| Fair | 0.8321 | 0.6892 | 0.6445 | 0.6661 |
| Private (ε=1.0) | 0.8234 | 0.6756 | 0.5987 | 0.6349 |
| Private+Fair | 0.8156 | 0.6623 | 0.6123 | 0.6364 |

### Fairness Improvements

| Metric | Base | Fair | Improvement |
|--------|------|------|-------------|
| Statistical Parity Difference | -0.36 | -0.14 | +0.22 |
| Disparate Impact | 0.64 | 0.86 | +0.22 |
| Equal Opportunity Difference | -0.18 | -0.08 | +0.10 |

### Privacy-Utility Trade-off

| Epsilon (ε) | Privacy Level | Accuracy Loss | Noise Level |
|-------------|---------------|---------------|-------------|
| 0.1 | Very High | -8.2% | High |
| 1.0 | Balanced | -3.1% | Medium |
| 5.0 | Low | -1.2% | Low |

## Documentation

### Additional Resources

- **`doc/planning_overview.md`**: Project workflow and team allocation strategy
- **`doc/explainability-llms.md`**: Design decisions for LLM interface
- **`project_statement.md`**: Original project requirements and objectives

### Key Concepts

**Differential Privacy**: Mathematical framework adding controlled noise to protect individual privacy while preserving statistical patterns.

**Fairness Mitigation**: Techniques to reduce algorithmic bias, including preprocessing (reweighing), in-processing, and post-processing methods.

**Explainable AI**: Methods to make machine learning models interpretable and trustworthy through various explanation techniques.

## Dependencies

### Core Libraries

- **Machine Learning**: `scikit-learn`, `xgboost`
- **Fairness**: `aif360`, `fairlearn`
- **Privacy**: Custom implementation using `numpy`
- **Explainability**: `shap`, `lime`, `omnixai`
- **LLM Interface**: `openai` (for LM Studio compatibility)
- **Data & Visualization**: `pandas`, `matplotlib`, `seaborn`

### Full Dependencies

See `requirements.txt` for complete list with versions. Key packages:

```
aif360==0.6.1          # Fairness metrics and algorithms
scikit-learn==1.7.2    # Machine learning framework
shap==0.49.1           # SHAP explanations
lime==0.2.0.1          # LIME explanations
omnixai==1.3.1         # Unified explainability framework
openai==2.8.1          # LLM interface (LM Studio compatible)
pandas==2.3.3          # Data manipulation
matplotlib==3.10.7     # Plotting
seaborn==0.13.2        # Statistical visualization
```

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-analysis`
3. **Make changes** and test thoroughly
4. **Update documentation** if needed
5. **Submit pull request** with clear description

### Code Standards

- **Consistent preprocessing**: Use the custom Adult dataset pipeline
- **Reproducible results**: Set random seeds (`np.random.seed(42)`)
- **Clear documentation**: Comment complex algorithms and decisions
- **Modular design**: Each notebook should be self-contained when possible

### Adding New Tasks

To extend the project:

1. **Follow naming convention**: `TaskName.ipynb`
2. **Load existing artifacts**: Reuse saved models and preprocessors
3. **Document approach**: Add explanation in docstring or markdown
4. **Update README**: Add task description and usage instructions

---

## License

This project is developed for educational purposes as part of the INFO-H420 course on Management of Data Science & Business Workflows.

## Acknowledgments

- **Dataset**: Adult Income Dataset from UCI Machine Learning Repository
- **Frameworks**: AIF360 (IBM), SHAP, LIME, OmniXAI
- **Course**: INFO-H420 - Management of Data Science & Business Workflows
- **Institution**: Université Libre de Bruxelles (ULB)

---

**Project Status**: Complete ✅  
**Last Updated**: December 2024  
**Team Size**: 6 members  
**Presentation**: December 12-15, 2024