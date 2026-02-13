# Contextual Multi-Armed Bandit for Personalized News Recommendation

## Overview

This project implements a Contextual Multi-Armed Bandit (CMAB) system for personalized news article recommendation. The system integrates supervised learning and reinforcement learning into a unified recommendation pipeline.

The objective is to recommend one of four news categories to users based on contextual information, while balancing exploration and exploitation.

The system consists of three core components:

- User context classification using supervised learning
- Contextual bandit algorithms for adaptive category selection
- End-to-end recommendation engine integrating classification and bandit decision-making

## Problem Statement

Traditional recommendation systems suffer from:

- The cold-start problem (lack of user history)
- Exploration–exploitation tradeoff in online learning

This project reformulates the recommendation task as a 12-arm contextual bandit problem, where:

- 3 user contexts × 4 news categories = 12 contextual arms
- Rewards are stochastic and context-dependent
- The objective is to maximize cumulative reward over time

### News Categories Considered

- Entertainment
- Education
- Technology
- Crime

Rewards are generated via a stochastic sampler to simulate real-world engagement noise.

## Context Classification

Three models were evaluated:

- Random Forest (100 trees, max depth 10)
- Logistic Regression (max_iter = 1000)
- Gradient Boosting (100 estimators, max depth 5)

**Validation accuracy:**

- Random Forest: 89%
- Gradient Boosting: 90%
- Logistic Regression: 87%

**Random Forest was selected due to:**

- Strong nonlinear modeling capability
- Robust generalization
- Best validation performance

**Note:** ~20% misclassification remains, meaning the bandit must operate under imperfect context predictions.

## Contextual Bandit Algorithms

### Algorithms Implemented

- Epsilon-Greedy
- Upper Confidence Bound (UCB)
- SoftMax

### Hyperparameter Tuning

Tuning was performed over 10,000 steps.

- **Epsilon-Greedy:** ε ∈ {0.01, 0.05, 0.1}
- **UCB:** exploration coefficient ∈ {0.5, 1.0, 2.0}
- **SoftMax:** τ = 1.0

### Results

- Moderate exploration performed best
- UCB with exploration coefficient = 1.0 achieved highest cumulative reward
- Very low exploration caused premature convergence
- Very high exploration reduced cumulative reward
- **UCB was selected for final deployment**

## End-to-End Recommendation Pipeline

For each test user:

1. Extract features
2. Predict context via Random Forest
3. Apply trained UCB policy to select category
4. Randomly sample article from selected category

### System Properties

- Modular architecture
- Sub-millisecond inference time
- Low memory footprint
- Easily scalable

## Performance Summary

### Classification

- ~90% validation accuracy

### Bandit Performance

- UCB achieved highest cumulative reward
- Clear context-dependent policy differentiation
- Balanced exploration-exploitation tradeoff


## Key Takeaways

- Structured exploration is critical in recommendation systems
- Context-aware policies outperform global strategies
- UCB provides efficient uncertainty-based exploration
- Contextual bandits bridge supervised learning and reinforcement learning
- Learned policies are interpretable and computationally efficient
