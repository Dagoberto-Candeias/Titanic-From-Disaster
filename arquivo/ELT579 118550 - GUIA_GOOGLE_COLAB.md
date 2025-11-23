# 🚀 Guia Completo - Google Colab

**Titanic: Machine Learning from Disaster - Versão Documentada**

---

## 📋 Opções de Execução no Colab

Você tem **3 opções** para executar o projeto no Google Colab:

### ✅ Opção 1: Notebook Jupyter (Recomendado)
### ✅ Opção 2: Script Python Completo
### ✅ Opção 3: Código Inline

---

## 🎯 Opção 1: Notebook Jupyter (Recomendado)

### Passo a Passo

#### 1. Abrir o Colab
- Acesse: https://colab.research.google.com/

#### 2. Fazer Upload do Notebook
```
File → Upload notebook → Escolher arquivo
Selecione: ELT579_118550_Titanic_Colab.ipynb
```

#### 3. Fazer Upload dos Dados
- Clique no ícone de pasta (📁) na barra lateral
- Clique em "Upload" (📤)
- Selecione `train.csv` e `test.csv`

#### 4. Executar as Células
```
Runtime → Run all
```

Ou execute célula por célula com `Shift + Enter`

#### 5. Baixar Resultados
- Execute a última célula (Download)
- Arquivo ZIP será baixado automaticamente
- Extraia e veja os resultados em `output/`

---

## 🔧 Opção 2: Script Python Completo

### Passo a Passo

#### 1. Criar Novo Notebook
```
File → New notebook
```

#### 2. Instalar Dependências
```python
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn python-docx
```

#### 3. Upload dos Arquivos
```python
from google.colab import files

# Upload train.csv e test.csv
print("📤 Faça upload de train.csv e test.csv")
uploaded = files.upload()

# Upload do script
print("📤 Faça upload de ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py")
script = files.upload()
```

#### 4. Executar o Script
```python
%run ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py
```

#### 5. Baixar Resultados
```python
import shutil
from google.colab import files

# Criar ZIP
shutil.make_archive('resultados', 'zip', 'output')

# Download
files.download('resultados.zip')
```

---

## 💻 Opção 3: Código Inline

### Passo a Passo

#### 1. Criar Novo Notebook

#### 2. Célula 1: Instalação
```python
%%capture
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn python-docx
```

#### 3. Célula 2: Upload de Dados
```python
from google.colab import files
uploaded = files.upload()
```

#### 4. Célula 3: Copiar e Colar
- Abra `ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py`
- Copie **TODO** o conteúdo
- Cole em uma nova célula
- Execute

#### 5. Célula 4: Download
```python
import shutil
from google.colab import files

shutil.make_archive('resultados', 'zip', 'output')
files.download('resultados.zip')
```

---

## 📊 Estrutura do Notebook

### Células Principais

```
┌─────────────────────────────────────┐
│ 1. Instalação de Dependências      │
│    !pip install ...                 │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 2. Upload de Dados                  │
│    files.upload()                   │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 3. Importações                      │
│    import pandas, numpy, etc        │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 4. Análise Exploratória (EDA)       │
│    9 visualizações                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 5. Feature Engineering              │
│    30+ features                     │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 6. Modelagem                        │
│    15+ algoritmos                   │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 7. Pipeline Completo                │
│    Execução integrada               │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 8. Resultados                       │
│    Relatórios + Gráficos            │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 9. Download                         │
│    files.download()                 │
└─────────────────────────────────────┘
```

---

## ⏱️ Tempo de Execução

### No Google Colab (GPU/TPU)

| Etapa | Tempo Estimado |
|-------|----------------|
| Instalação | 1-2 min |
| Upload | 30 seg |
| EDA | 2-3 min |
| Feature Engineering | 1-2 min |
| Modelagem | 8-12 min |
| Relatórios | 1 min |
| **TOTAL** | **15-20 min** |

### Dicas para Acelerar:
- Use GPU: `Runtime → Change runtime type → GPU`
- Reduza número de modelos (comente alguns)
- Reduza folds de validação cruzada

---

## 📁 Arquivos Gerados no Colab

Após execução, você terá:

```
output/
├── submission_titanic_final.csv         (10 KB)
├── graficos/
│   ├── 01_eda_completa.png             (500 KB)
│   ├── 02_comparacao_modelos.png       (200 KB)
│   └── 03_matriz_confusao.png          (150 KB)
└── relatorios/
    ├── RELATORIO_FINAL_TITANIC.md      (30 KB)
    ├── RELATORIO_FINAL_TITANIC.docx    (50 KB)
    └── resultados_modelos.csv          (2 KB)

TOTAL: ~1 MB
```

---

## 🎨 Visualização no Colab

### Gráficos Inline

Os gráficos aparecerão diretamente no notebook:

```python
# Gráfico será exibido automaticamente
plt.show()
```

### Tabelas Formatadas

```python
# Usar display() para melhor visualização
display(df.head())
display(resultados_tabela)
```

### Métricas em Tempo Real

```python
# Prints coloridos e formatados
print("✅ Etapa concluída!")
print(f"📊 Acurácia: {accuracy:.4f}")
```

---

## 💡 Dicas e Truques

### 1. Salvar Progresso
```python
# Salvar checkpoint
import pickle

with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Carregar depois
with open('checkpoint.pkl', 'rb') as f:
    pipeline = pickle.load(f)
```

### 2. Montar Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Salvar resultados no Drive
!cp -r output/ /content/drive/MyDrive/Titanic/
```

### 3. Ver Arquivos Gerados
```python
# Listar arquivos
!ls -lh output/
!ls -lh output/graficos/
!ls -lh output/relatorios/

# Ver conteúdo
!cat output/relatorios/RELATORIO_FINAL_TITANIC.md
```

### 4. Baixar Arquivo Específico
```python
from google.colab import files

# Baixar apenas submissão
files.download('output/submission_titanic_final.csv')

# Baixar apenas relatório
files.download('output/relatorios/RELATORIO_FINAL_TITANIC.docx')
```

### 5. Executar em Background
```python
%%capture output
# Código que demora muito
# Será executado sem mostrar prints
```

---

## 🔍 Troubleshooting

### Problema 1: "ModuleNotFoundError"
**Solução:**
```python
!pip install --upgrade nome-do-modulo
```

### Problema 2: "FileNotFoundError: train.csv"
**Solução:**
```python
# Verificar se arquivo foi carregado
!ls -la

# Fazer upload novamente
from google.colab import files
uploaded = files.upload()
```

### Problema 3: Memória Insuficiente
**Solução:**
```python
# Usar GPU
# Runtime → Change runtime type → GPU

# Ou reduzir dados
train_sample = train.sample(frac=0.5, random_state=42)
```

### Problema 4: Execução Muito Lenta
**Solução:**
```python
# Reduzir número de modelos
# Comente alguns modelos no código

# Reduzir folds
cv = StratifiedKFold(n_splits=3)  # ao invés de 5
```

### Problema 5: Sessão Desconectada
**Solução:**
```python
# Salvar progresso regularmente
import pickle

# Após cada etapa importante
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(dados, f)
```

---

## 📊 Exemplo de Execução Completa

### Código Completo para Colab

```python
# ============================================================================
# TITANIC - EXECUÇÃO COMPLETA NO GOOGLE COLAB
# ============================================================================

# 1. INSTALAÇÃO
!pip install -q pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn python-docx

# 2. UPLOAD DE DADOS
from google.colab import files
print("📤 Faça upload de train.csv e test.csv")
uploaded = files.upload()

# 3. UPLOAD DO SCRIPT
print("📤 Faça upload de ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py")
script = files.upload()

# 4. EXECUTAR
print("🚀 Executando análise completa...")
%run ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py

# 5. DOWNLOAD
import shutil
print("📦 Preparando download...")
shutil.make_archive('titanic_resultados', 'zip', 'output')
files.download('titanic_resultados.zip')

print("✅ Concluído!")
```

---

## 🎯 Checklist de Execução

### Antes de Executar
- [ ] Conta Google ativa
- [ ] Acesso ao Google Colab
- [ ] Arquivos train.csv e test.csv baixados
- [ ] Arquivo .py ou .ipynb pronto

### Durante a Execução
- [ ] Dependências instaladas
- [ ] Dados carregados
- [ ] Script executando sem erros
- [ ] Gráficos sendo gerados
- [ ] Prints aparecendo

### Após a Execução
- [ ] Pasta output/ criada
- [ ] 3 gráficos gerados
- [ ] 2 relatórios gerados
- [ ] Arquivo de submissão criado
- [ ] Download realizado
- [ ] ZIP extraído localmente

---

## 📈 Monitoramento da Execução

### Ver Progresso em Tempo Real

```python
# Prints informativos aparecem automaticamente
# Exemplo de saída:

"""
================================================================================
🚀 EXECUTANDO ANÁLISE COMPLETA DO TITANIC
================================================================================

📊 ETAPA 1: ANÁLISE EXPLORATÓRIA
   ✅ 9 visualizações criadas
   ✅ Gráfico salvo: output/graficos/01_eda_completa.png

🛠️ ETAPA 2: FEATURE ENGINEERING
   ✅ 25 features criadas
   ✅ Valores ausentes tratados

🤖 ETAPA 3: MODELAGEM
   [1/19] ✅ RandomForest: 0.8350 ± 0.0167
   [2/19] ✅ XGBoost: 0.8401 ± 0.0145
   [3/19] ✅ LightGBM: 0.8380 ± 0.0156
   ...

🎉 ANÁLISE COMPLETA FINALIZADA COM SUCESSO!
"""
```

---

## 🔗 Links Úteis

### Google Colab
- **Colab:** https://colab.research.google.com/
- **Documentação:** https://colab.research.google.com/notebooks/
- **Atalhos:** Ctrl+M H (mostrar atalhos)

### Kaggle
- **Competição:** https://www.kaggle.com/c/titanic
- **Submissão:** https://www.kaggle.com/c/titanic/submit
- **Leaderboard:** https://www.kaggle.com/c/titanic/leaderboard

### Recursos
- **Pandas:** https://pandas.pydata.org/docs/
- **Scikit-learn:** https://scikit-learn.org/
- **XGBoost:** https://xgboost.readthedocs.io/

---

## 🎓 Dicas Avançadas

### 1. Usar TPU (Mais Rápido)
```python
# Runtime → Change runtime type → TPU
import tensorflow as tf
print("TPU disponível:", tf.config.list_physical_devices('TPU'))
```

### 2. Aumentar RAM
```python
# Runtime → Change runtime type → High-RAM
import psutil
print(f"RAM disponível: {psutil.virtual_memory().total / 1e9:.2f} GB")
```

### 3. Executar em Lote
```python
# Executar múltiplas configurações
configs = [
    {'n_estimators': 100, 'max_depth': 5},
    {'n_estimators': 200, 'max_depth': 10},
    {'n_estimators': 300, 'max_depth': 15}
]

for config in configs:
    print(f"Testando: {config}")
    # Executar com config
    # Salvar resultados
```

### 4. Comparar Versões
```python
# Salvar resultados de cada execução
import json
from datetime import datetime

resultado = {
    'timestamp': datetime.now().isoformat(),
    'acuracia': melhor_acuracia,
    'modelo': melhor_modelo,
    'config': config
}

with open(f'resultado_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(resultado, f)
```

---

## ✅ Resumo Final

### Opção Recomendada: Notebook Jupyter

**Vantagens:**
- ✅ Execução célula por célula
- ✅ Visualização inline
- ✅ Fácil debug
- ✅ Markdown explicativo
- ✅ Salva progresso

**Passos:**
1. Upload do `.ipynb`
2. Upload dos dados
3. Run all
4. Download resultados

**Tempo:** 15-20 minutos

---

**Pronto para usar no Google Colab! 🚀**

*Última atualização: 09/10/2025 12:52*
