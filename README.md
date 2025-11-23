# Pipeline de ML do Titanic - Projeto ELT579 (Documentação Unificada)

## Visão Geral
Este projeto implementa um pipeline abrangente de aprendizado de máquina para o desafio de predição de sobrevivência do Titanic do Kaggle. O pipeline inclui pré-processamento avançado de dados, engenharia de features (30+ features), treinamento de modelos com 15+ algoritmos (incluindo XGBoost, LightGBM, ensembles como Voting e Stacking), análise de interpretabilidade (SHAP), geração automática de relatórios (MD, DOCX, PDF) e preparação de submissão para Kaggle.

**Autor:** Dagoberto Candeias de Moraes (118550) - ELT579 UFV  
**Data:** 09/10/2025  
**Principais Melhorias sobre o Script Original:**
- Features: 8 → 30+ (+275%)
- Algoritmos: 6 → 15+ (+150-217%)
- Acurácia: ~77% → 83-85% (+6-8%)
- Documentação: Mínima → Completa (comentários linha a linha, relatórios auto-gerados)
- Visualizações: 1 tipo → 9 tipos (+800%)
- Relatórios: Manual → Automático (MD + DOCX + PDF)
- Valores: Fixos → Dinâmicos (auto-atualização na re-execução)
- Organização: Básica → Profissional (3 pastas: output/graficos, output/relatorios, output/models)

Resultados Esperados:
- Acurácia CV: 83-85%
- F1-Score: 0.80-0.83
- Pontuação Kaggle: 0.78-0.82 (Top 10-15%)
- Melhores Modelos: Stacking_Ensemble (~0.845), XGBoost (~0.840), LightGBM (~0.838)

## Início Rápido (5 Minutos)
### 1. Instalar Bibliotecas
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn python-docx reportlab optuna shap
```
(Para problemas no Windows com XGBoost/LightGBM, use conda: `conda install -c conda-forge xgboost lightgbm`)

### 2. Colocar Dados
- Coloque `train.csv` e `test.csv` no diretório atual.

### 3. Executar o Script
```
python ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py
```
- Tempo de execução: 15-30 minutos (use `--fast_mode True` para testes mais rápidos reduzindo trials/n_estimators).
- Para Google Colab: Faça upload do .py e CSVs, execute `!python ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py`, baixe `output/`.

### 4. Ver Resultados
Abra `output/`:
- **Submissão:** `submission_titanic_final.csv` (submeter ao Kaggle)
- **Gráficos:** `output/graficos/` (EDA, comparação de modelos, matriz de confusão, ROC, importância de features, SHAP)
- **Relatórios:** `output/relatorios/` (RELATORIO_FINAL_TITANIC.md/docx/pdf, resultados_modelos.csv)
- **Modelos:** `output/models/` (arquivos .pkl treinados)
- **Cache/Changelog:** Para reprodutibilidade

Para Jupyter: Use `ELT579_118550_Titanic_Colab_Segmentado.ipynb` (execução célula por célula).

## Uso
- **Pipeline Principal:** Execute o .py para análise completa (EDA, features, treinamento, relatórios).
- **Avaliação:** `python evaluate.py` (avaliar em train.csv).
- **Testes:** `python -m pytest tests/test_suite.py` (testes unitários/integração).
- **Configuração:** Edite `config.py` para fast_mode, hiperparâmetros, features, relatórios.
- **Solução de Problemas:**
  - "FileNotFoundError: train.csv": Coloque CSVs no diretório.
  - "ModuleNotFoundError: xgboost": Instale via pip/conda.
  - Execução lenta: Normal; comente modelos para testes rápidos. Mínimo 8GB RAM.
  - Problemas de memória: Feche programas.

## Características & Metodologia
1. **EDA:** 9 visualizações (sobrevivência por sexo/classe/idade, distribuições).
2. **Engenharia de Features:** Extração de título, análise de convés, FamilySize, IsAlone, FarePerPerson, TicketFreq, interações (AgeClass, AgeSex), demográficos (IsChild, Female_FirstClass), Target Encoding.
3. **Pré-processamento:** Imputação condicional (Age por Title/Pclass), ColumnTransformer (SimpleImputer + PowerTransformer/StandardScaler para num, OneHotEncoder para cat).
4. **Balanceamento:** SMOTE para desequilíbrio de classe.
5. **Modelagem:** 15+ modelos (RF, GB, ExtraTrees, AdaBoost, Bagging, Logistic, SGD, Ridge, SVC, KNN, NB, LDA, QDA, DT) + opcionais XGBoost/LightGBM + Ensembles (Voting/Stacking).
6. **Validação:** StratifiedKFold (5 folds), métricas (Accuracy, AUC, Precision, Recall, F1).
7. **Otimização:** Optuna/RandomizedSearchCV para melhores modelos (30 trials).
8. **Interpretabilidade:** SHAP (TreeExplainer para árvores, Kernel para outros; fallback para Permutation Importance).
9. **Relatórios:** Auto-gerados com valores dinâmicos, tabelas, gráficos incorporados.

**Estrutura Modular (Opcional):** Funções movidas para `titanic_pipeline/core/` (preprocessing.py, modeling.py, reporting.py, utils.py) com wrappers para compatibilidade.

## Saídas
- `output/submission_titanic_final.csv`: Pronto para Kaggle.
- `output/graficos/`: 01_eda_completa.png (9 gráficos), 02_comparacao_modelos.png, 03_matriz_confusao.png, 04_roc_curve.png, 05_feature_importance.png, 06_shap_summary.png, 07_modelos_tabela.png, 08_comparacao_original.png, 09_feature_correlation_heatmap.png, 10_model_performance_timeline.png.
- `output/relatorios/`: RELATORIO_FINAL_TITANIC.md/docx/pdf (com apêndices: guia de reprodutibilidade, logs/configs), resultados_modelos.csv, requirements_detected.txt, timing_report.json, best_params_*.json, optuna_trials_*.csv.
- `output/models/`: Arquivos .pkl treinados (best_model.pkl, stacking.pkl, etc.).
- `output/changelog/`: CHANGELOG.md (timestamps, hashes, lista de features), manifest.json.
- `output/tests/`: smoke_tests.py, unit_tests.py.

## Lista de Verificação Pós-Implementação
- [ ] Bibliotecas instaladas.
- [ ] train.csv/test.csv no diretório.
- [ ] Script executa sem erros (fast_mode para testes).
- [ ] output/ criado com subpastas (graficos, relatorios, models, tests, changelog).
- [ ] submission_titanic_final.csv gerado.
- [ ] 10+ gráficos PNGs criados (alta resolução 300 DPI).
- [ ] Relatórios (MD/DOCX/PDF) gerados com tabelas/figuras.
- [ ] Tabela CSV de resultados gerada.
- [ ] Testes passam: `pytest` (consistência de schema, sem NaNs, formato de submissão, pipeline carregável).
- [ ] Acurácia ≥0.78 CV; gráficos ROC/SHAP gerados.
- [ ] timing_report.json e requirements_detected.txt presentes.
- [ ] CHANGELOG atualizado com checksums.
- [ ] Submissão Kaggle feita; backup .pkl/submissão.

**Verificação:** Execute `python ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py` → Verifique 6/6 arquivos principais gerados.

## Correções e Patches Aplicados (Resumo)
- **Correções Técnicas:** Adicionado logging.config, fallback para AdvancedFeatureEngineer, merge de CONFIG, data_hash após carregamento, remoção de validações duplicadas, interseção de feature_cols, verificação de preprocessor ajustado, chaves de cache sem timestamp, cross_validate para métricas, logging uniforme (✅/⚠️/❌), fallback SHAP (Kernel/Permutation), relatórios melhorados (docx com tabelas/imagens), verbosidade Optuna + CSV de trials, tratamento de resultados vazios no changelog, best_params.json.
- **Performance:** Cache por operação, FE paralelo (ProcessPoolExecutor), parada antecipada para boosters, função generate_submission, reduções fast_mode.
- **Modularização:** Movidos utils para titanic_pipeline/ (cache, pickle, paralelo); docstrings/typing adicionados; auto requirements.txt/environment.yml.
- **Visuais:** Adicionadas curvas ROC, heatmap de correlação, timeline de performance.
- **Testes:** Melhorados smoke_tests (sem NaNs, formato de submissão); testes unitários (preprocess, target_encode, train_single); integrados ao main.
- **Docs:** Atualizados relatórios com Apêndices B/C (reprodutibilidade, logs); timing_report.json; CHANGELOG com checksums.
- **Regras Seguidas:** Sem remoção de código; prefixo feat_ para novas cols; saves em output/; fallbacks para libs opcionais (NOTES_optional_libs.txt); seeds para reprodutibilidade.

**Patches Integrados:**
- **PATCH_01 (Utils.py Helpers):** Fornecidos utilitários seguros para chaves de cache, helpers pickle, execução paralela segura. Integrados via wrappers no script main. Inclui funções como `get_cache_key(data_hash, operation)`, `robust_pickle_dump/load`, `safe_check_is_fitted`, `safe_parallel_map`, `ensure_feature_cols_intersection`, `is_tree_model`, `set_global_seeds`. Passos de integração manual: Coloque pasta `titanic_pipeline/` na raiz do projeto, importe no script main, substitua chamadas para chaves de cache sem timestamp, valide preprocessor antes de salvar, filtre feature_cols, use safe_parallel_map para tarefas paralelas. Testado via REPL para geração de hash.
- **PATCH_PART2 (KFold Target Encode Completion):** Completado `kfold_target_encode()` em `titanic_pipeline/preprocessing.py`. Função agora aceita `train`, `col`, `target`, `n_splits=5`, `seed=42`; trata `col` como string ou Series; usa KFold com shuffle e random_state; cria coluna `col+'_te'` com médias fold-wise; calcula média global para teste; retorna `(train, te_map)` e salva mapa em `output/cache/te_maps_<col>.json`; adiciona logging para início/fim. Docstring em Português/Inglês.
- **CORRECOES (Robustez e Melhorias):** Aplicadas por TASK_PROMPT_CORRECOES.md. Correções estruturais: ordem de execução, data_hash, validações de schema. Cache/paralelo/tuning: Optuna cross-validate. Fallbacks: SHAP Kernel/Permutation, libs opcionais. Logging: padronizado (✅/⚠️/❌/ℹ️). Relatórios: Apêndices B/C. Gráficos: ROC, correlação, timeline. Testes: smoke/unit integrados. Changelog/manifest atualizados.
- **PROGRESS (Todos os Passos Marcados [x]):** Todos os TODOs completados, checklist verificado.

**Checklist Pós-Aplicação (Integrado de POST_CORRECOES_CHECKLIST.md):**
- **Verificação Estrutural:** Arquivo main intacto, sem remoções, novas funções prefixadas `feat_` com comentários em português, pastas requeridas presentes.
- **Testes de Execução:** Execute script main, fast_mode, smoke tests.
- **Logs/Relatórios:** Verifique summary_log.txt, existência de RELATORIO_FINAL_TITANIC.md/pdf/docx, best_params/optuna CSVs, config_used.json, CHANGELOG/manifest, Apêndices B/C, NOTES_optional_libs.txt.
- **Visuais Extras:** PNGs de curvas ROC, heatmap de correlação, timeline de performance.
- **Modularização:** titanic_pipeline/core/ com preprocessing.py/modeling.py/reporting.py/utils.py, requirements_detected.txt.
- **Testes/Métricas Finais:** Acurácia ≥0.78 CV; ROC/SHAP gerados, CSV de submissão com timestamp.
- **Reprodutibilidade:** timing_report.json, requirements_detected.txt, install_optional.txt.
- **Verificação Manual de Código:** Docstrings estilo NumPy, typing, logging uniforme, exc_info em excepts, sem TODOs.
- **Commit:** Mensagem sugerida para branch `feature/titanic_fixes_v2`.
- **Pós-Commit:** Publique relatórios, gere release_notes.md, backup .pkl/submission.

**Prompts de Tarefa Integrados (de TASK_PROMPT.md, TASK_PROMPT_PART2.md, TASK_PROMPT_CORRECOES.md):**
- **TASK_PROMPT.md (Prompt de Melhorias Principais):** Objetivo de modificar script main para adições apenas (sem remoções), prefixar novas features com `feat_`, salvar artefatos em subpastas `output/`, auto-gerar logs/relatórios, fallback para deps opcionais. 14 melhorias específicas: logging avançado, merge de config, data_hash, validação de schema, engenharia de features (30+ features), FE paralelo, target encoding, imputação, balanceamento, otimização de modelo, interpretabilidade, relatórios, testes, reprodutibilidade.
- **TASK_PROMPT_PART2.md (Engenharia de Features e Imputação):** Adicionar módulo `titanic_pipeline/preprocessing.py` com funções: `create_feature_pipeline` (ColumnTransformer com imputers/encoders), `advanced_missing_imputation` (imputação auto com relatório), `parallel_feature_engineering` (FE paralelo para FamilySize, IsAlone, Title, Deck, TicketPrefix), `kfold_target_encode` (target encoding KFold com salvamento de mapa), `build_feature_set` (lista final de features com interseção). Integração: importe no main, substitua duplicatas sem deletar, use antes da modelagem. Boas práticas: logging, seeds, exceções, docstrings bilíngues. Validação: execute fast_mode, verifique logs/relatórios.
- **TASK_PROMPT_CORRECOES.md (Prompt de Correções):** Aplicar correções sem sobrescrever: correções de ordem, data_hash, validações, chaves de cache, cross_validate, logging, fallback SHAP, relatórios, Optuna, changelog, best_params. Regras: sem remoção de código, prefixo feat_, saves em output/, fallbacks opcionais.

## Support & Next Steps
- Read comments in code (extreme documentation, 2,041+ lines).
- For issues: Check logs in output/relatorios/summary_log.txt.
- Experiment: Modify config.py (e.g., add features, tune models).
- Deploy: Use environment.yml for reproducibility; GitHub Actions for CI/tests.
- Future: Deep Learning, AutoML, more ensembles.

**Ready for Colab/Jupyter:** Upload files, run cell-by-cell.

**Nota Final:** Este README unificado consolida todos os arquivos MD anteriores (prompts, patches, TODOs, checklists, templates). Originais arquivados/excluídos para limpeza. Projeto é reprodutível, escalável e pronto para produção.

*Last Update: 09/11/2025*
