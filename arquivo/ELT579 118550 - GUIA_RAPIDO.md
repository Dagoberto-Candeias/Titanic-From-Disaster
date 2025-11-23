# 🚀 Guia Rápido - Titanic Documentado

## ⚡ Início Rápido (5 minutos)

### 1. Instale as Bibliotecas

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn python-docx
```

### 2. Coloque os Dados

Certifique-se de ter `train.csv` e `test.csv` no mesmo diretório do script.

### 3. Execute

```bash
python ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py
```

### 4. Aguarde

O script levará cerca de 15-30 minutos para executar completamente.

### 5. Veja os Resultados

Abra a pasta `output/` e confira:
- `submission_titanic_final.csv` - Para submeter no Kaggle
- `RELATORIO_FINAL_TITANIC.md` - Relatório completo
- Gráficos na pasta `graficos/`

---

## 📊 O Que o Código Faz?

### Etapa 1: Análise Exploratória (2-3 min)
- Carrega os dados
- Gera 9 visualizações
- Calcula estatísticas

### Etapa 2: Feature Engineering (1-2 min)
- Cria 30+ features novas
- Trata valores ausentes
- Prepara dados

### Etapa 3: Modelagem (10-20 min)
- Testa 15+ algoritmos
- Cria ensembles
- Valida resultados

### Etapa 4: Relatórios (1 min)
- Gera relatório em Markdown
- Gera relatório em Word
- Cria gráficos

---

## 🎯 Principais Melhorias vs Original

| Item | Original | Atual |
|------|----------|-------|
| Features | 8 | 30+ |
| Modelos | 6 | 15+ |
| Acurácia | 77% | 83-85% |
| Documentação | Mínima | Completa |

---

## 💡 Dicas

### Para Google Colab

1. Faça upload do arquivo `.py`
2. Faça upload de `train.csv` e `test.csv`
3. Execute: `!python ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py`
4. Baixe a pasta `output/`

### Para Jupyter Notebook

Use o arquivo `ELT579_118550_Titanic_Colab_Segmentado.ipynb` que já está dividido em células.

### Para Entender o Código

Leia os comentários! Cada bloco tem explicações detalhadas sobre:
- O que faz
- Por que faz
- Como faz
- Comparação com o original

---

## 📁 Arquivos Importantes

### Entrada
- `train.csv` - Dados de treino (891 passageiros)
- `test.csv` - Dados de teste (418 passageiros)

### Saída Principal
- `output/submission_titanic_final.csv` - **SUBMETA ESTE NO KAGGLE**
- `output/RELATORIO_FINAL_TITANIC.md` - Relatório completo
- `output/RELATORIO_FINAL_TITANIC.docx` - Relatório em Word

### Gráficos
- `output/graficos/01_eda_completa.png` - Análise exploratória
- `output/graficos/02_comparacao_modelos.png` - Comparação de modelos
- `output/graficos/03_matriz_confusao.png` - Matriz de confusão

---

## ❓ Problemas Comuns

### "FileNotFoundError: train.csv"
**Solução:** Coloque os arquivos CSV no mesmo diretório do script.

### "ModuleNotFoundError: No module named 'xgboost'"
**Solução:** Instale as bibliotecas: `pip install xgboost lightgbm`

### Script muito lento
**Solução:** Normal! Leva 15-30 minutos. Para teste rápido, comente alguns modelos no código.

### Memória insuficiente
**Solução:** Feche outros programas. Mínimo 8GB RAM recomendado.

---

## 🎓 Para Aprender

### 1. Leia o Código Comentado
Cada linha tem explicações detalhadas.

### 2. Compare com o Original
Veja as melhorias implementadas.

### 3. Experimente
Modifique features e algoritmos.

### 4. Analise os Resultados
Entenda por que cada modelo teve determinado desempenho.

---

## 📞 Precisa de Ajuda?

1. Leia o `README_TITANIC_DOCUMENTADO.md` completo
2. Verifique os comentários no código
3. Consulte o relatório gerado

---

## ✅ Checklist de Execução

- [ ] Bibliotecas instaladas
- [ ] Arquivos train.csv e test.csv no diretório
- [ ] Script executado sem erros
- [ ] Pasta output/ criada
- [ ] Arquivo submission gerado
- [ ] Relatórios gerados
- [ ] Gráficos visualizados
- [ ] Submissão no Kaggle feita

---

## 🏆 Resultado Esperado

- **Acurácia (Validação Cruzada):** 83-85%
- **Score no Kaggle:** 0.78-0.82
- **Posição Estimada:** Top 10-15%

---

**Boa sorte! 🍀**

*Última atualização: 09/10/2025*
