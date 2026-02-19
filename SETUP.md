# Configuração do Ambiente de Desenvolvimento

Este guia passo a passo explica como configurar o ambiente de desenvolvimento no seu novo computador.

## ⚠️ IMPORTANTE: Compatibilidade NumPy 2.x

Este projeto usa **NumPy 2.x** com `conda-forge` para garantir compatibilidade binária.
Se você receber um erro como `ImportError: numpy.core.multiarray failed to import`, veja [FIX_NUMPY_COMPATIBILITY.md](FIX_NUMPY_COMPATIBILITY.md).

## Pré-requisitos

- Anaconda ou Miniconda instalado
- VSCode instalado (opcional)

## Passo 1: Criar o Ambiente Conda

### Opção A: Usando o script automatizado (recomendado)

```
bash
.\setup_environment.bat
```

### Opção B: Manual

```
bash
conda env create -f environment.yml
conda activate titanic_ml
```

## Passo 2: Configurar VSCode

1. Abra o VSCode
2. Vá para `File > Open Folder` e selecione a pasta do projeto
3. O VSCode deve automaticamente detectar o ambiente conda `titanic_ml`
4. Se não detectar, pressione `Ctrl+Shift+P` e digite `Python: Select Interpreter`
5. Selecione `titanic_ml` da lista

### Configurações já aplicadas (via .vscode/settings.json)

- Python interpreter configurado para o ambiente conda
- Linting com flake8 habilitado
- Formatting com black habilitado
- Testing com pytest habilitado

## Passo 3: Verificar a Instalação

Execute o script de teste:

```
bash
python src/test_imports.py
```

## Comandos Úteis

### Ativar o ambiente
```
bash
conda activate titanic_ml
```

### Desativar o ambiente
```
bash
conda deactivate
```

### Listar ambientes
```
bash
conda env list
```

### Remover ambiente (se necessário)
```
bash
conda env remove -n titanic_ml
```

## Executando o Projeto

### Gerar relatório
```
bash
python src/gerar_relatorio_titanic.py
```

### Executar testes
```
bash
pytest tests/
```

### Treinar modelo
```
bash
python scripts/train.py
```

## Solução de Problemas

### VSCode não detecta o ambiente conda

1. Certifique-se de que o Anaconda está no PATH do sistema
2. Ou configure o caminho do conda nas configurações do VSCode:
   - Vá para `File > Preferences > Settings`
   - Procure por `Python > Conda Path`
   - Adicione o caminho para o executável do conda (ex: `C:\Users\seu_usuario\anaconda3\Scripts\conda.exe`)

### Erro de permissão ao executar scripts

Execute o PowerShell como administrador e defina a política de execução:
```
powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problemas com bibliotecas

Reinstale as dependências:
```
bash
conda activate titanic_ml
pip install -r requirements.txt
