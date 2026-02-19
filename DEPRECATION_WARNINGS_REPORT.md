# 🔍 Relatório de Deprecation Warnings - Análise Completa

**Data**: 2026-02-19  
**Projeto**: Titanic ML Pipeline  
**Branch**: fix/preprocessing-no-import-side-effects

---

## 📊 Sumário

| Categoria | Severity | Ocorrências | Impacto |
|-----------|----------|-------------|---------|
| **1. sklearn.experimental.enable_iterative_imputer** | 🟠 Medium | 2 | Será removido em sklearn 1.5+ |
| **2. pd.util.hash_pandas_object (deprecado em pandas 2.0)** | 🔴 High | 1 | Pode quebrar, usar hashlib direto |
| **3. .values (numpy array implícito)** | 🟡 Low | 15+ | Preferir .to_numpy() ou .array |
| **4. XGBoost use_label_encoder=False** | 🟡 Low | 1 | Deprecado, remover |
| **5. Filtros de Warning globais** | 🟠 Medium | 2 | Suprimir específicos, não "ignore all" |

**Total de Issues**: ~25 correções necessárias  
**Tempo estimado para corrigir**: 1-2 horas

---

## 🔴 PRIORIDADE 1: CRITICAL (Quebrará em futuro próximo)

### 1.1 `pd.util.hash_pandas_object` ❌ DEPRECATED em Pandas 2.0+

**Arquivo**: `titanic_pipeline/core/utils.py:112`  
**Código Atual**:
```python
def data_hash_calc(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
```

**Problema**: 
- `pd.util.hash_pandas_object` foi movido/deprecado
- Pode não estar mais disponível em pandas 2.2+
- Gera warning durante importação

**Solução**:
```python
def data_hash_calc(df: pd.DataFrame) -> str:
    """Calcula hash de um DataFrame usando hashlib."""
    import io
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    content = buffer.getvalue().encode()
    return hashlib.md5(content).hexdigest()
```

**Impacto**: 🔴 Crítico - Quebra cache em pandas 2.2+

---

## 🟠 PRIORIDADE 2: HIGH (Deprecação ativa)

### 2.1 `sklearn.experimental.enable_iterative_imputer` ⚠️ EXPERIMENTAL API

**Arquivos**: 
- `titanic_pipeline/preprocessing.py:7`
- `titanic_pipeline/core/preprocessing.py:9`

**Código Atual**:
```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
```

**Problema**:
- `enable_iterative_imputer` é API experimental
- Será removido em sklearn 1.5
- Gera FutureWarning

**Solução**:
```python
# REMOVER ambas as linhas - IterativeImputer é officialmente stable em sklearn 1.2+
# Não precisa do enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

**Impacto**: 🟠 Alta - Quebra em sklearn 1.5

---

## 🟡 PRIORIDADE 3: MEDIUM (Deprecação suave)

### 3.1 `.values` ao invés de `.to_numpy()` ou `.array`

**Ocorrências identificadas**:
```
titanic_pipeline/core/pipeline.py:81 - X_train.values
titanic_pipeline/core/pipeline.py:83 - y_train.values
titanic_pipeline/core/pipeline.py:112 - X_test.values
titanic_pipeline/core/reporting.py:163 - y_train.values
titanic_pipeline/features/engineer.py:46 - data["Survived"].values
titanic_pipeline/features/engineer.py:69 - data[feature_cols].values
titanic_pipeline/features/preprocessing.py:82 - X_df.iloc[:, top_idx].values
titanic_pipeline/preprocessing.py:747 - mapped.values
titanic_pipeline/core/utils.py:112 - (já coberto acima)
... ~15+ ocorrências
```

**Problema**:
- `.values` pode retornar tipos inesperados (object array ao invés de numpy array)
- `.to_numpy()` é mais explícito e previsível
- Gera FutureWarning em pandas 2.0+

**Solução exemplo** (usar em .values → .to_numpy()):
```python
# ANTES
X_train = X_train.values

# DEPOIS
X_train = X_train.to_numpy(dtype=np.float64, na_value=np.nan)
```

**Impacto**: 🟡 Médio - Pode causar comportamento inesperado

**Linha 163 em reporting.py** (mais específico):
```python
# ANTES
y_vals = y_train.values if hasattr(y_train, "values") else y_train

# DEPOIS
if hasattr(y_train, "to_numpy"):
    y_vals = y_train.to_numpy(dtype=np.int64)
else:
    y_vals = np.asarray(y_train, dtype=np.int64)
```

---

### 3.2 XGBoost `use_label_encoder=False` ⚠️ DEPRECATED

**Arquivo**: `titanic_pipeline/features/preprocessing.py:66`

**Código Atual**:
```python
model = XGBClassifier(
    n_estimators=100, 
    random_state=42, 
    use_label_encoder=False,  # ← DEPRECATED
    verbosity=0
)
```

**Problema**:
- `use_label_encoder` foi removido em XGBoost 1.5+
- Não é mais necessário
- Gera DeprecationWarning

**Solução**:
```python
model = XGBClassifier(
    n_estimators=100, 
    random_state=42,
    # use_label_encoder removido - não é mais necessário
    verbosity=0
)
```

**Impacto**: 🟡 Baixo - Código ignora o parâmetro

---

## 🟢 PRIORIDADE 4: SOFT DEPRECATIONS (Boas práticas)

### 4.1 Filtros Globais `warnings.filterwarnings("ignore")`

**Arquivos**: 
- `src/train.py:7-9`

**Código Atual**:
```python
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```

**Problema**:
- Suprime TODOS os warnings dessas categorias
- Mascara problemas reais
- Dificulta debug

**Solução**:
```python
# Suprimir warnings ESPECÍFICOS apenas
import warnings

# Para sklearn.experimental
warnings.filterwarnings(
    "ignore", 
    message=".*enable_iterative_imputer.*",
    category=FutureWarning
)

# Para bottleneck (se ainda houver)
warnings.filterwarnings(
    "ignore",
    message=".*bottleneck.*",
    category=UserWarning
)
```

**Impacto**: 🟢 Baixo - Melhora visibilidade de erros

---

## 📋 Plano de Ação

### Fase 1: CRÍTICO (Data: 2026-02-19)
- [ ] **Fix**: `pd.util.hash_pandas_object` → hashlib direto
  - Arquivo: `titanic_pipeline/core/utils.py:112`
  - Tempo: 10 min

### Fase 2: ALTO (Data: 2026-02-19)
- [ ] **Remove**: `from sklearn.experimental import enable_iterative_imputer`
  - Arquivos: 2 (preprocessing.py, core/preprocessing.py)
  - Tempo: 5 min
  
- [ ] **Replace**: `.values` → `.to_numpy()` (top 5)
  - Arquivos principais:
    - `titanic_pipeline/core/pipeline.py` (3 ocorrências)
    - `titanic_pipeline/core/reporting.py:163` (1 ocorrência)
  - Tempo: 15 min

- [ ] **Remove**: `use_label_encoder=False` XGBoost
  - Arquivo: `titanic_pipeline/features/preprocessing.py:66`
  - Tempo: 2 min

### Fase 3: MÉDIO (Data: próxima PR)
- [ ] Replace remaining `.values` → `.to_numpy()` (batch de 10+)
  - Tempo: 30 min
  - Arquivos: múltiplos

### Fase 4: BOAS PRÁTICAS (Data: próxima PR)
- [ ] Refinar `warnings.filterwarnings()` em `src/train.py`
  - Tempo: 10 min

---

## 🧪 Como Testar

```bash
# 1. Ativar o ambiente titanic_ml
conda activate titanic_ml

# 2. Rodar com warnings ativos
python -W default src/gerar_relatorio_titanic.py 2>&1 | grep -i "deprecat\|future\|pending"

# 3. Rodar testes
pytest tests/ -W default::DeprecationWarning -W default::FutureWarning
```

---

## 📚 Referências

- [Pandas 2.0 Migration Guide](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html)
- [scikit-learn Experimental APIs](https://scikit-learn.org/stable/modules/preprocessing.html)
- [XGBoost Changelog](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
- [NumPy 2.0 dtype migration](https://numpy.org/devdocs/release/2.0.0-notes/migration_guide.html)

---

## 📊 Estatísticas Código

| Métrica | Valor |
|---------|-------|
| Total warnings identificados | 5 categorias |
| Total ocorrências | ~25 |
| Arquivos afetados | 8+ |
| Severidade máxima | CRITICAL (1 issue) |
| Tempo total para fix | ~1.5-2 horas |

