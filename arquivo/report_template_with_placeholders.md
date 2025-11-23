# Relatório - Titanic (Template com Placeholders)

## 1) Introdução

... 

## 2) Objetivo

... 

## 3) Metodologia

Descreva: EDA, Feature Engineering, Pré-processamento, Modelos testados, Validação.

## 4) Resultados

### 4.1 Scores (CV)

| Modelo | CV Mean Accuracy | CV Std |

|---|---:|---:|

| RandomForest |  |  |

| GradientBoosting |  |  |

| Stacking |  |  |


### 4.2 Métricas no conjunto de treino

| Métrica | Valor |

|---|---:|

| accuracy |  |

| f1 |  |

| precision |  |

| recall |  |


### 4.3 Figuras

![](./output/figs/dist_survived.png)

![](./output/figs/surv_by_pclass.png)

![](./output/figs/surv_by_sex.png)

![](./output/figs/age_dist.png)

![](./output/figs/confusion_matrix_train.png)

![](./output/figs/feature_importances.png)


### 4.4 Print do Kaggle

![](./output/figs/kaggle_score.png)


## 5) Observações e Próximos Passos

- Tentar LightGBM / XGBoost.
- Otimizar hiperparâmetros (Optuna / BayesOpt).
- Analisar interpretabilidade (SHAP).
