Resumo das alterações (branch: fix/format-lint)

Objetivo
- Corrigir, robustecer, otimizar e documentar o script principal do projeto Titanic.
- Eliminar erros de lint (ruff/flake8), garantir testes automatizados e limpar arquivos legados.

Principais mudanças aplicadas
1. Correções de estilo e lint
   - Aplicado reflow para eliminar E501 (linhas muito longas) em arquivos grandes.
   - Reordenados imports e removidos imports no meio do arquivo (E402).
   - Corrigido shadowing/local import issues (F823) e pequenos problemas de indentação.

2. Robustez e refatoração
   - Criação/ajustes em helpers de engenharia de features (AdvancedFeatureEngineer).
   - Localização de imports pesados (xgboost, lightgbm, shap, imblearn) e flags de disponibilidade.
   - Introdução de alias local para importações pontuais (ex.: _RandomForestClassifier) para evitar conflitos.
   - Externalizado conteúdo narrativo grande para `arquivo/RELATORIO_TEMPLATE.md`.

3. Testes e CI
   - Suíte pytest incluída/ajustada: todos os testes locais passaram (7 passed, warnings informativos do sklearn).
   - Workflow GitHub Actions (run tests) foi adicionado/atualizado (conforme commit anterior).

4. Limpeza de ruído/arquivamento
   - Movi versões antigas e notebooks para `arquivo/legacy/` preservando histórico (git mv).
   - Adicionei um changelog/manifest e artefatos de relatório em `arquivo/`.

Resultados verificados
- Ruff (lint) no arquivo alterado: limpo (E501/E402/F823 resolvidos).
- Pytest: 7 passed, 6 warnings (informativos).
- Commit e push realizados para `fix/format-lint`.

Arquivos adicionados/alterados (resumo)
- arquivo/titanic_profissionalizado_v4.0/ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio_FIXED.py (correções)
- arquivo/RELATORIO_TEMPLATE.md (conteúdo externalizado)
- arquivo/PR_DESCRIPTION_UPDATE.md (este arquivo)
- diversas movimentações para `arquivo/legacy/` (git mv preservando histórico)

Recomendações / próximos passos
- Revisão rápida dos blocos de visualização/EDA e geração de relatórios (mudanças foram majoritariamente mecânicas).
- (Opcional) Atualizar a descrição do PR no GitHub com o conteúdo deste arquivo (posso fazer se der autorização API/token).
- (Opcional) Remover/arquivar definitivamente ou compactar `arquivo/legacy/` em release/zip para liberar espaço.

Como atualizar a descrição do PR
- Manual: copie o conteúdo deste arquivo e cole na caixa de descrição do PR no GitHub (PR já existente: "style: apply black + ruff fixes").
- Automático: me autorize a usar um token GitHub (não recomendado aqui) ou eu posso dar os passos/uma PR body sugerida para você colar.

Se quiser, atualizo a descrição do PR automaticamente (opção avançada) ou crio um comentário no PR com este resumo. O que prefere agora?
- (A) Eu atualizo a descrição do PR manualmente (você autoriza o uso de token/API)
- (B) Eu crio um comentário no PR com este resumo (faço automaticamente)
- (C) Nada — apenas mantemos o arquivo no repositório para referência

Assinatura: alterações feitas pelo pipeline de correção na branch `fix/format-lint`.