# Resolvendo Incompatibilidades de NumPy 2.x

## Problema Identificado

Há um **conflito de compatibilidade binárias** no seu ambiente Python base:
- NumPy 2.2.6 está instalado (versão recente)
- Mas SciPy, CatBoost, Bottleneck foram compilados para **NumPy 1.x**
- Resultado: `ImportError: numpy.core.multiarray failed to import`

## Solução: Usar o Ambiente Conda `titanic_ml`

O arquivo `environment.yml` foi **atualizado** para resolver isso automaticamente:

### 1. Criar/Atualizar o Ambiente

```bash
# Opção A: Criar novo (se não existe)
conda env create -f environment.yml

# Opção B: Atualizar (se já existe)
conda env update -f environment.yml --name titanic_ml
```

### 2. Ativar o Ambiente

**Windows (PowerShell/CMD):**
```bash
conda activate titanic_ml
```

**Mac/Linux:**
```bash
source activate titanic_ml
```

### 3. Verificar que Funciona

```bash
python test_warnings_check.py
```

Isso mostrará informações de versão sem erros.

## Por que `conda-forge` é Melhor que `pip`

| Aspecto | conda-forge | pip |
|---------|------------|-----|
| **Binários** | Compilados para NumPy 2.x | Podem ser NumPy 1.x |
| **Compatibilidade** | Testados juntos | Combinação aleatória |
| **Velocidade** | Mais rápido (pré-compilado) | Compilação local |
| **SciPy/SHAP** | Funciona sem problemas | Pode falhar |

## Detalhes Técnicos

O `environment.yml` agora especifica:

```yaml
channels:
  - conda-forge    # ← Pacotes compilados para NumPy 2.x
  - defaults       # ← Fallback

dependencies:
  - numpy>=2.0,<3  # ← Permite 2.0+
  - pandas>=2.1    # ← Suporta NumPy 2.x
  - scipy          # ← Compilado para NumPy 2.x
  - scikit-learn   # ← Funciona
  - shap           # ← Sem problemas de ABI
```

## Se Ainda Receber Erros

### Erro: "Module compiled for NumPy 1.x"

```bash
# Remova o ambiente antigo
conda env remove -n titanic_ml

# Recrie do zero
conda env create -f environment.yml --force
```

### Erro: "ImportError" ao importar SciPy

```bash
# Dentro do ambiente titanic_ml, force rebuild
pip install --force-reinstall numpy scipy --no-cache-dir
```

### Erro: "bottleneck" não encontrado

Já está no `environment.yml` via pandas. Se persistir:

```bash
conda install bottleneck -c conda-forge
```

## Verificação Final

Depois de ativar `titanic_ml`, tipo:

```bash
python -c "import sklearn, scipy, shap; print('All imports OK')"
```

Se não houver erros, está pronto para usar!

## Próximos Passos

Com o ambiente funcionando corretamente:

1. **Rodar testes:**
   ```bash
   pytest -q
   ```

2. **Gerar relatório:**
   ```bash
   python src/gerar_relatorio_titanic.py
   ```

3. **Fazer predições:**
   ```bash
   python scripts/prever_passageiro.py
   ```
