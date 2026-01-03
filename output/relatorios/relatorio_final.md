# Titanic ML Pipeline - Relatório Final

## Resumo Executivo

- **Total de Modelos Treinados:** 15
- **Total de Features:** 23
- **Melhor Acurácia:** 0.8283

## Resultados dos Modelos

| Modelo | Acurácia Média | Desvio Padrão | Melhor Score |
|--------|---------------|---------------|--------------|
| SVC | 0.8283 | 0.0120 | 0.8384 |
| RidgeClassifier | 0.8227 | 0.0032 | 0.8249 |
| AdaBoost | 0.8182 | 0.0126 | 0.8316 |
| LinearSVC | 0.8171 | 0.0151 | 0.8350 |
| LogisticRegression | 0.8159 | 0.0151 | 0.8350 |
| CatBoost | 0.8103 | 0.0278 | 0.8316 |
| MLPClassifier | 0.8092 | 0.0199 | 0.8249 |
| RandomForest | 0.8047 | 0.0291 | 0.8316 |
| LightGBM | 0.8047 | 0.0153 | 0.8215 |
| GradientBoosting | 0.8025 | 0.0177 | 0.8182 |
| XGBoost | 0.7980 | 0.0167 | 0.8114 |
| Bagging | 0.7957 | 0.0297 | 0.8215 |
| ExtraTrees | 0.7823 | 0.0231 | 0.8148 |
| DecisionTree | 0.7609 | 0.0055 | 0.7677 |
| SGDClassifier | 0.7508 | 0.0795 | 0.8215 |

## Features Utilizadas

1. Pclass
2. Sex
3. Age
4. SibSp
5. Parch
6. Fare
7. Embarked
8. FamilySize
9. IsAlone
10. TicketPrefix
11. feat_AgeBin
12. feat_FareBin
13. feat_AgeCategory_v2
14. feat_FareCategory_v2
15. feat_Age_missing
16. feat_Cabin_missing
17. feat_Embarked_missing
18. feat_Fare_missing
19. feat_Pclass_te
20. feat_Sex_te
21. feat_Embarked_te
22. feat_Title_te
23. feat_TicketPrefix_te

## Configuração Utilizada

```json
{}
```
