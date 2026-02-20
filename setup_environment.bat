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
    echo ERRO ao criar ambiente com environment.yml. Tentando alternativa com conda-forge...
    REM Instala pacotes criticos via conda para garantir binarios compativeis
    conda create -n titanic_ml python=3.11 numpy pandas scipy scikit-learn matplotlib seaborn -c conda-forge -y
    call conda activate titanic_ml
    echo Instalando dependencias restantes com pip...
    pip install -r requirements.txt
    echo Instalando bibliotecas adicionais para relatorios...
    pip install python-docx fpdf2
)

echo.
echo 2. Ativando ambiente...
call conda activate titanic_ml

echo.
echo 3. Validando o ambiente com o script de verificação...
python scripts\validate_environment.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ----------------------------------------------------------------------
    echo AVISO: A validação do ambiente falhou.
    echo Algumas bibliotecas podem estar com versões incorretas ou faltando.
    echo Verifique o log acima para mais detalhes.
    echo ----------------------------------------------------------------------
    echo.
)

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
