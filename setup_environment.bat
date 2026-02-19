@echo off
REM =============================================================================
REM Script de Setup - Titanic ML Project
REM Cria o ambiente conda e instala as dependências
REM =============================================================================

echo ========================================
echo Configurando ambiente Titanic ML...
echo ========================================

REM Verifica se o conda está instalado
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Conda não encontrado! Por favor, instale o Anaconda ou Miniconda.
    pause
    exit /b 1
)

echo.
echo 1. Criando ambiente conda...
conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo ERRO ao criar ambiente. Tentando alternativa...
    conda create -n titanic_ml python=3.11 -y
    call conda activate titanic_ml
    pip install -r requirements.txt
    pip install python-docx fpdf
)

echo.
echo 2. Ativando ambiente...
call conda activate titanic_ml

echo.
echo 3. Verificando instalação...
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, xgboost, lightgbm, catboost, optuna, pytest; print('Todas as bibliotecas instaladas com sucesso!')"

echo.
echo ========================================
echo Ambiente configurado com sucesso!
echo ========================================
echo.
echo Para ativar o ambiente no futuro, use:
echo   conda activate titanic_ml
echo.
echo Para executar o projeto, use:
echo   python src/gerar_relatorio_titanic.py
echo.

pause
