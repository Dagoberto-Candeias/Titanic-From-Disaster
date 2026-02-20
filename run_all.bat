@echo off
REM =============================================================================
REM Script de ExecuÃ§Ã£o Completa - Titanic ML Project
REM Executa limpeza, validaÃ§Ã£o, linting, testes e o pipeline completo em sequÃªncia.
REM =============================================================================

echo ========================================
echo  INICIANDO EXECUCAO COMPLETA DO PROJETO
echo ========================================

REM --- Verificacao do Diretorio de Execucao ---
if not exist "environment.yml" (
    echo.
    echo ERRO CRITICO: O script deve ser executado a partir da raiz do projeto.
    echo Arquivo 'environment.yml' nao encontrado no diretorio atual.
    echo Diretorio atual: %CD%
    goto :error
)

REM --- AtivaÃ§Ã£o do Ambiente ---
echo.
echo [PASSO 1/6] Ativando ambiente conda 'titanic_ml'...
call conda activate titanic_ml
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Nao foi possivel ativar o ambiente 'titanic_ml'.
    echo Execute 'setup_environment.bat' primeiro.
    goto :error
)
echo Ambiente ativado com sucesso.

REM --- Limpeza ---
echo.
echo [PASSO 2/6] Executando script de limpeza...
python scripts/limpar_projeto.py
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: A limpeza do projeto falhou.
    goto :error
)
echo Limpeza concluida.

REM --- ValidaÃ§Ã£o do Ambiente ---
echo.
echo [PASSO 3/6] Validando o ambiente de execucao...
python scripts/validate_environment.py
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: A validacao do ambiente falhou. Verifique as dependencias.
    goto :error
)
echo Ambiente validado com sucesso.

REM --- VerificaÃ§Ã£o de Qualidade de CÃ³digo (Linting) ---
echo.
echo [PASSO 4/6] Verificando a qualidade do codigo (Linting)...
python scripts/check_code_quality.py
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Foram encontrados problemas de qualidade de codigo.
    goto :error
)
echo Qualidade do codigo verificada com sucesso.

REM --- Testes Automatizados ---
echo.
echo [PASSO 5/6] Executando testes automatizados (pytest)...
pytest -n auto
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Os testes falharam. Abortando execucao do pipeline.
    goto :error
)
echo Testes aprovados com sucesso.

REM --- ExecuÃ§Ã£o do Pipeline Completo ---
echo.
echo [PASSO 6/6] Executando o pipeline completo de analise e treinamento...
python run_full_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: A execucao do pipeline principal falhou. Verifique o log em 'logs/'.
    goto :error
)
echo Pipeline executado com sucesso.

echo.
echo =================================================================
echo  ðŸŽ‰ PROCESSO COMPLETO FINALIZADO COM SUCESSO!
echo  Verifique a pasta 'output/' para os relatorios e modelos.
echo =================================================================
goto :end

:error
echo.
echo =================================================================
echo  âŒ FALHA NA EXECUCAO!
echo  O processo foi interrompido devido a um erro no passo anterior.
echo  Verifique as mensagens de erro acima para mais detalhes.
echo =================================================================

:end
echo.
pause