# INFO-H420 — Management of Data Science & Business Workflows  

## Project: Responsible Data Science

---

> Goal  
> Combine responsible data science principles to study an ML pipeline. You'll work with the Adult dataset and train classifiers to predict income (>50K). The project explores classification, fairness, privacy, explainability, and LLM-based explanation interfaces.

---

## Table of Contents

- [INFO-H420 — Management of Data Science \& Business Workflows](#info-h420--management-of-data-science--business-workflows)
  - [Project: Responsible Data Science](#project-responsible-data-science)
  - [Table of Contents](#table-of-contents)
  - [1. Classification](#1-classification)
  - [2. Fairness](#2-fairness)
  - [3. Privacy](#3-privacy)
  - [4. Privacy and Fairness](#4-privacy-and-fairness)
  - [5. Explainability](#5-explainability)
  - [6. Explainability and LLMs](#6-explainability-and-llms)
  - [7. Free Exploration](#7-free-exploration)
  - [Instructions \& Deadlines](#instructions--deadlines)

---

## 1. Classification

- Preprocess the Adult dataset and binarize Age.
- Split the data into train / validation / test sets.
- Train a classifier (referred to as *the classifier*).
- Evaluate and report the classifier's performance on the test set.

---

## 2. Fairness

- Assume protected attributes are **Age** and **Sex**.
- Assess the group fairness of the classifier.
- Choose a fairness metric and apply a mitigation technique to obtain a *fair classifier*.
- Report the chosen fairness metric measured on both the classifier and the fair classifier.

---

## 3. Privacy

- Treat **Age** and **Sex** as sensitive attributes.
- Compute a cross-tabulation for combinations of these sensitive attributes.
- Apply local differential privacy to participants' responses for Age and Sex; produce a private dataset (explore multiple ε values).
- Compute a cross-tabulation on the private dataset and estimate group counts; quantify estimation errors.
- Split the private dataset similarly to (1), train a classifier (the *private classifier*), and report its performance.
- Analyze whether privacy affects model performance compared to the original classifier.

---

## 4. Privacy and Fairness

- Use the same fairness metric and mitigation method from section 2.
- Create a fairness-adjusted model using the private dataset (the *private+fair classifier*).
- As an auditor with access to the true Age and Sex, measure fairness of the private+fair classifier using the real protected values.
- Compare fairness of the private+fair classifier to the fair classifier and draw conclusions.

---

## 5. Explainability

- Study explainability for the *private classifier*.
- Find instances where the model is wrong but highly confident; provide explanations for those cases.
- With access to true Age and Sex, investigate whether noisy (private) versions of these attributes explain the model's confident mistakes.

---

## 6. Explainability and LLMs

- Choose an explainability method.
- Build a natural language interface to present explanations (e.g., feature importances or example-based explanations) in human-friendly text.
- You may use local small LLMs (LM Studio is suggested) to generate the natural language descriptions.

---

## 7. Free Exploration

- Explore the dataset for additional insights (responsible data science themes).
- Report any interesting findings.

---

## Instructions & Deadlines

- This project is worth 30% of the course grade (6/20).
- Work in groups of six — use "Group Choice for Project" on Université Virtuelle (UV). If you can't find a partner, post in the UV discussion forum.
- Deliverables: a short report describing your solutions, choices, and assumptions, plus any supporting files.
- Upload to "Project" on UV by **December 8, 2025**.
- Prepare an 8-minute presentation for a 20-minute slot on **December 12 or December 15** — choose your preferred date using the "Select Day for Project Presentation" link.

> Note: LM Studio may help for section 6: <https://lmstudio.ai>
