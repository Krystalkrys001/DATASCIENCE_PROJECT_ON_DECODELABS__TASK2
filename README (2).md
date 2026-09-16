# Project 2: Supervised Learning, Fraud Detection Pipeline
**DecodeLabs Data Science Industrial Training, 2026 Batch**

## Overview

This project builds and compares two classification models to detect fraudulent transactions in a real-world, highly imbalanced dataset of 284,807 credit card transactions, where only 0.17% (492 transactions) are actually fraud.

The core challenge isn't writing model code, it's handling extreme class imbalance correctly, avoiding data leakage at every step, and choosing evaluation metrics that actually mean something on skewed data.

## The Core Problem

A model that predicts "legitimate" for every single transaction, doing zero actual work, scores **99.83% accuracy** while catching **0 of 492 real fraud cases**. This project treats that as the starting warning, not a footnote: accuracy is actively misleading here, and every decision below is built around metrics that can't be gamed the same way.

## 1. Train/Test Split, Before Any Resampling

Split 80/20 using stratify=y, which preserves the 0.17% fraud ratio identically in both the training set (0.1729%) and test set (0.1720%). This matters because a random split without stratification risks concentrating rare fraud cases unevenly across the two sets, distorting everything downstream.

## 2. Handling Class Imbalance: SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraud examples by interpolating between real fraud cases nearest neighbors, rather than simply duplicating them. Applied **only to the training set**, after the split, never before. Applying it before splitting would leak synthetic copies of test-set fraud into training, producing test results that look great but don't reflect real-world performance.

**A second, less obvious leakage risk**: SMOTE's "nearest neighbor" calculation is distance-based. Amount ranges up to 25,691 in this dataset while the anonymized V1-V28 features mostly sit between -60 and 10. Without scaling first, Amount would dominate every distance calculation, meaning the synthetic fraud examples get built almost entirely on price similarity, not on the more informative but smaller-scale patterns in the other 28 features. For the Logistic Regression pipeline, features were scaled with StandardScaler **before** SMOTE was applied, for exactly this reason.

## 3. Two Models, Two Different Requirements

| Model | Scaling Required? | Why |
|---|---|---|
| Logistic Regression | Yes | Distance/coefficient-based, sensitive to feature scale |
| Random Forest | No | Splits features ordinally (threshold-based), immune to scale |

Both models were trained on SMOTE-balanced training data, then evaluated on the **real, untouched, imbalanced test set** (56,864 legitimate vs. 98 fraud), since that's what reflects actual production conditions.

## 4. Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Recall (fraud) | 91.84% | 85.71% |
| Precision (fraud) | 5.78% | 62.69% |
| False alarms | 1,467 | 50 |
| Fraud missed | 8 of 98 | 14 of 98 |
| ROC-AUC | 0.9708 | 0.9745 |

**Confusion matrix, Logistic Regression:**
[[55397  1467]
 [    8    90]]

**Confusion matrix, Random Forest:**
[[56814    50]
 [   14    84]]

## 5. Decision: Random Forest

Both models achieve near-identical, strong ROC-AUC scores (~0.97), meaning both are genuinely capable of separating fraud from legitimate transactions overall. The real difference is in the precision/recall trade-off at the decision threshold:

Logistic Regression catches slightly more fraud (92% vs. 86%) but is wrong 94% of the time it raises an alarm, 1,467 innocent transactions flagged out of 1,557 total alerts. Random Forest sacrifices 6 percentage points of recall for a jump to 63% precision, cutting false alarms by nearly 97%, from 1,467 down to 50.

**Random Forest is the better model for this use case.** In a real fraud operations team, a model with a 94% false-alarm rate trains investigators to distrust and eventually ignore its flags entirely, which defeats the model's purpose regardless of how many true positives it catches on paper.

## Tech Stack

Python, Pandas, Scikit-learn, imbalanced-learn (SMOTE)

## Files

- project2_fraud_detection.py — full pipeline: load, split, scale, SMOTE, train, evaluate, compare
- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud), 284,807 transactions

## Key Takeaway

The hardest part of this project wasn't the algorithms, it was resisting the two traps built into the data itself: accuracy that lies by default, and a resampling step that silently breaks if scaling happens in the wrong order. Getting the pipeline right mattered more than which model got picked.

LINKR FROM MY LINKEDIN POST ON TASK 2 - https://lnkd.in/p/eTmNpqXq
