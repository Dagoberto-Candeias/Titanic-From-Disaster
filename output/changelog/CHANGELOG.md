# Changelog - Titanic ML Pipeline

## Versão Atual - 2025-11-22

### Melhorias Implementadas
- ✅ K-Fold Target Encoding para Title_Group, TicketPrefix, Deck, Embarked
- ✅ Missingness indicators (feat_*_missing)
- ✅ Bins e categorizações (feat_AgeBin, feat_FareBin, etc.)
- ✅ Imputação avançada com validação
- ✅ Seleção de features via modelo
- ✅ Ensemble stacking
- ✅ Calibração sistemática
- ✅ Importância de permutação
- ✅ Tuning automatizado (Optuna + RandomizedSearchCV)
- ✅ Testes smoke
- ✅ Versionamento automático
- ✅ Reprodutibilidade com datahash
- ✅ Relatórios aprimorados
- ✅ Modo seguro com verificações de libs

### Estatísticas do Pipeline
- **Features criadas:** 19
- **Modelos treinados:** 29
- **Tempo total:** 15485.77s
- **Melhor acurácia:** 0.8533

### Arquivos Gerados
- output/submission_titanic_final.csv
- output/models/best_model_pipeline.pkl
- output/relatorios/RELATORIO_FINAL_TITANIC.md
- output/changelog/CHANGELOG.md
- output/changelog/manifest.json
