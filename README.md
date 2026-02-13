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

## Dataset Description

### News Dataset

- **Total articles:** 208,083
- **Original categories:** 42 (HuffPost categories)
- **Mapped to 4 target categories:**
  - Education (7 original categories)
  - Crime (2 categories)
  - Technology (3 categories)
  - Entertainment (30 categories)

**Category distribution:**

- Entertainment ≈ 73%
- Education ≈ 20%
- Technology ≈ 5%
- Crime ≈ 3%

This imbalance reflects realistic content distributions in media platforms.

## Feature Engineering

### User Features

- Age (missing values handled via median imputation)
- Browser version
- Region code
- Subscription status

### Preprocessing Steps

- Label encoding for categorical variables
- Combined vocabulary fitting to avoid train-test mismatch
- 80-20 stratified train-validation split

**Dataset sizes:**

- 2,000 labeled users (training)
- 2,000 unlabeled users (test)

## Context Classification

Three models were evaluated:

- Random Forest (100 trees, max depth 10)
- Logistic Regression (max_iter = 1000)
- Gradient Boosting (100 estimators, max depth 5)

**Validation accuracy:**

- Random Forest: 78–82%
- Gradient Boosting: 75–78%
- Logistic Regression: 68–72%

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
- **UCB:** exploration coefficient ∈ {0.5, 1.0, 3.0}
- **SoftMax:** τ = 1.0

### Results

- Moderate exploration performed best
- UCB with exploration coefficient = 1.0 achieved highest cumulative reward
- Very low exploration caused premature convergence
- Very high exploration reduced cumulative reward
- **UCB was selected for final deployment**

## Arm-Level Reward Learning

The system learned differentiated rewards across the 12 arms:

- Some arms achieved strong positive average rewards (6–9 range)
- Some arms were consistently negative (-1 to -7)
- Some were near zero (uncertain or weak preference)
- The algorithm naturally concentrated recommendations on high-performing arms while maintaining minimal exploration

This confirms successful learning of context-specific reward structure.

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

- ~80% validation accuracy

### Bandit Performance

- UCB achieved highest cumulative reward
- Clear context-dependent policy differentiation
- Balanced exploration-exploitation tradeoff

### Efficiency

- Constant-time arm selection
- Minimal computation overhead
- Suitable for real-time applications

## Baseline Comparisons

Compared to:

### Random Policy
- Achieves average corpus reward
- Significantly underperforms learned policies

### Global Greedy Policy (Context-Agnostic)
- Ignores user segmentation
- Misses context-specific optimal arms

### Content-Based Filtering
- Requires heavy feature engineering
- Does not inherently manage exploration

**Contextual bandits provide superior structured learning under uncertainty.**

## Limitations

- Offline evaluation using simulated rewards (no real user feedback)
- Imperfect context classification affects recommendation accuracy
- Static context assumption (no drift adaptation)
- Small action space (12 arms)
- Single-objective reward (engagement only)

## Future Improvements

- Online A/B testing with real users
- Scalable contextual linear or neural bandits
- Multi-objective optimization (diversity, revenue, fairness)
- Drift-aware retraining mechanisms
- Diversity-constrained recommendation policies

## Key Takeaways

- Structured exploration is critical in recommendation systems
- Context-aware policies outperform global strategies
- UCB provides efficient uncertainty-based exploration
- Contextual bandits bridge supervised learning and reinforcement learning
- Learned policies are interpretable and computationally efficient

## Conclusion

This project demonstrates a complete, modular implementation of a contextual multi-armed bandit recommendation system. By integrating supervised context classification with reinforcement learning-based decision-making, the system successfully learns personalized, context-dependent policies under uncertainty.

The implementation provides a strong foundation for scaling toward production-grade adaptive recommendation systems.