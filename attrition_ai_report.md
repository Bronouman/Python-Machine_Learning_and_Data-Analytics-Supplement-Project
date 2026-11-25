# Employee Attrition AI Report

## Executive Summary
Employee attrition significantly differs by marital status (chi-square p = 7.53e-07), with single employees exhibiting the highest attrition rate at 22.22% compared to 13.05% for married and 12.26% for divorced staff. In contrast, tests for distance from home (Mann-Whitney p = 0.065) and ANOVA (p = 0.233) show no statistically significant relationship with attrition at alpha = 0.05. Evaluated models show Decision Tree achieving an AUC of 0.530 and Neural Network achieving an AUC of 0.548, meaning predictive power is weak across all tested models.

## Key Drivers
- Marital status is a statistically significant driver of attrition (chi-square p = 7.527457319553302e-07), where single employees leave at a rate of 22.22%, whereas married and divorced employees leave at rates of 13.05% and 12.26% respectively.

## Recommended Model
Neither

## Confidence Level
Low

## HR Action Items
1. Design targeted retention strategies and engagement initiatives specifically aimed at single employees, who experience an attrition rate of 22.22%.
2. Do not rely on the evaluated Decision Tree (AUC 0.530) or Neural Network (AUC 0.548) for individual attrition risk forecasting until stronger predictors and models are developed.
3. Refrain from making policy decisions based on distance from home alone, as its association with attrition is not statistically significant at alpha = 0.05 (p = 0.065).
