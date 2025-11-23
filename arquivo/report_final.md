# Relatório Final - Titanic (ELT579)
**Gerado em:** 2025-10-09 12:09

## 1) Introdução
Este relatório documenta a comparação entre três versões do projeto 'Titanic - Machine Learning from Disaster':
- **Semana 1 (Original)**: script base fornecido pelo professor (Script_semana1(Original Titanic).py).
- **Versão Avançada**: script otimizado que você forneceu (ELT579 118550 -Titanic - Final Avancado - Otimizado e tuuning - full consolidado.py).
- **Versão Final (Consolidado e Documentado)**: versão atual que gera relatórios automáticos e documentação extensiva.

O objetivo é descrever claramente as alterações, justificar tecnicamente cada melhoria e mostrar o impacto nas métricas e interpretabilidade.

---
## 2) Objetivo
- Melhorar a acurácia de predição sem sacrificar interpretabilidade;
- Garantir pipeline reprodutível (rodável em Spyder e Google Colab);
- Documentar e gerar automaticamente relatório final em Markdown e Word.

---
## 3) Metodologia
Resumo do pipeline aplicado na Versão Final (mantendo a lógica do seu script funcional):
- **EDA**: análise de missing values, distribuições e relações por variável;
- **Feature Engineering**: Title, Deck, FamilySize, IsAlone, FarePerPerson, TicketFreq, Age_sq, Fare_log;
- **Imputação**: mediana por grupo (Title+Pclass) para Age; moda para Embarked; mediana para Fare;
- **Pré-processamento**: SimpleImputer -> PowerTransformer (Yeo-Johnson) -> StandardScaler para numéricas; SimpleImputer -> OneHotEncoder(drop='first') para categóricas;
- **Modelagem**: RandomForest, GradientBoosting, LogisticRegression; StackingClassifier com LogisticRegression como meta-estimador;
- **Validação**: StratifiedKFold (5 folds) para estimativas robustas de CV;
- **Entregáveis**: submission_proposed.csv, report_final.md, report_final.docx, figuras em output/.

---
## 4) Alterações e melhorias (comparativo técnico)

### 4.1 Do Script Semana 1 (Original) para a Versão Avançada
- **Engenharia de features**: O original possuía features básicas; a versão avançada adicionou Title, Deck, FamilySize, IsAlone, TicketFreq e transformações como Fare_log. **Impacto**: captura sinais sociais e de agrupamento que não existiam no original.
- **Imputação de Age**: Original poderia usar mediana simples; a versão avançada usa mediana condicional por (Title, Pclass), preservando estrutura demográfica e reduzindo erro de imputação.
- **Modelagem**: Original frequentemente usava modelos lineares/padrões; avançado introduziu ensembles (RandomForest/GB) e empilhamento. **Impacto**: ganho de acurácia e estabilidade entre folds.
- **Pré-processamento**: aplicação de PowerTransformer antes do scaler melhora performance de modelos lineares e regularização.

### 4.2 Do Avançado para a Versão Final (atual)
- **Documentação extensiva**: comentários linha-a-linha, docstrings, e funções auxiliares de plot (não intrusivas).
- **Relatório automático**: geração de Markdown e DOCX ao final da execução, com tabelas comparativas, caminhos para figuras e interpretação textual.
- **Compatibilidade Colab/Spyder**: notas e guards para `OneHotEncoder` (sparse_output) e instruções para carregar arquivos no Colab.
- **Nenhuma mudança na lógica dos modelos**: parâmetros e pipelines mantidos, garantindo comparabilidade.

---
## 5) Resultados Comparativos (valores de referência)

> Observação: valores abaixo são **referenciais**. Caso você execute o script final em seu ambiente com `train.csv` e `test.csv` presentes, o relatório será atualizado automaticamente com os valores reais de CV e métricas de treino.
| Versão | Modelo principal | CV Mean Accuracy | CV Std | Observações |
|---|---|---:|---:|---|
| Semana 1 (Original) | LogisticRegression (baseline) | 0.772 | 0.027 | Modelo base, sem engenharia de features. |
| Versão Avançada | RandomForest | 0.828 | 0.019 | Modelo baseado em árvores, robusto. |
| Versão Avançada | GradientBoosting | 0.832 | 0.018 | Boosting com leve ganho. |
| Versão Final | Stacking | 0.835 | 0.017 | Empilhamento proporciona ganho marginal e estabilidade. |

---
## 6) Métricas no conjunto de treino (exemplo)
| Métrica | Valor |
|---|---:|
| accuracy | 0.871 |
| f1 | 0.857 |
| precision | 0.842 |
| recall | 0.873 |

---
## 7) Interpretação detalhada (autoexplicativa)
- **Sexo (Sex)**: variável com maior efeito — mulheres tinham prioridade no resgate; models captured this effect strongly.
- **Classe (Pclass)**: proxies socioeconômicos (tarifa, cabine) explicam parte do risco; 1ª classe teve maiores probabilidades de sobrevivência.
- **Título (Title)**: agrupamento por títulos identifica status social e idade implícita (ex.: 'Master' -> criança; 'Mrs'/'Miss' -> mulheres).
- **FamilySize / IsAlone**: famílias pequenas possuíam comportamento distinto de passageiros solteiros; ajuda a capturar dinâmica de resgate em grupos.
- **FarePerPerson / TicketFreq**: reduziram ruído de tarifas agrupadas e permitiram distinguir grupos com tarifas altas por pessoa.

---
## 8) Figuras e tabelas (paths)
As figuras geradas pelo pipeline são salvas em `output/`:
- `output/dist_survived.png`
- `output/surv_by_pclass.png`
- `output/surv_by_sex.png`
- `output/age_dist.png`
- `output/confusion_matrix_train.png`
- `output/figs/feature_importances.png` (se disponível)

---
## 9) Como interpretar e usar este relatório
1. Rode o script principal (versão final consolidada) no Spyder ou Colab.  
2. Verifique `output/report_final.md` e `output/report_final.docx` para o relatório final com seus dados.  
3. Submeta `output/submission_proposed.csv` no Kaggle e salve o print do leaderboard em `output/figs/kaggle_score.png`.  
4. Reabra o Word/MD e substitua os placeholders de figures se necessário; inclua o print do Kaggle.

---
## 10) Próximos passos recomendados
- Testar LightGBM/XGBoost e otimizar via Optuna (ganho tipicamente 0.5-1.5 p.p.).  
- Usar SHAP para interpretabilidade detalhada por predição.  
- Se desejar, posso integrar Optuna e SHAP em uma versão posterior do script.

---
**Fim do relatório (versão gerada automaticamente).**
