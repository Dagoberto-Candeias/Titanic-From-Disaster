@echo off
REM =============================================================================
REM Script de Testes - Titanic ML Project
REM Executa os testes unitarios usando pytest com as configuracoes do pyproject.toml
REM Gera relatorios de cobertura no terminal e em XML (coverage.xml).
REM =============================================================================

echo ========================================
echo  INICIANDO BATERIA DE TESTES AUTOMATIZADOS
echo ========================================

REM --- Verificacao do Diretorio ---
if not exist "pyproject.toml" (
    echo.
    echo ERRO: O script deve ser executado a partir da raiz do projeto.
    echo Arquivo 'pyproject.toml' nao encontrado.
    goto :error
)

REM --- AtivaÃ§Ã£o do Ambiente ---
echo.
echo [PASSO 1/2] Ativando ambiente conda 'titanic_ml'...
call conda activate titanic_ml
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Nao foi possivel ativar o ambiente 'titanic_ml'.
    echo Execute 'setup_environment.bat' primeiro.
    goto :error
)
echo Ambiente ativado.

REM --- ExecuÃ§Ã£o dos Testes ---
echo.
echo [PASSO 2/2] Rodando Pytest...
echo (As configuracoes de cobertura e relatorios sao lidas do pyproject.toml)
pytest -n auto

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FALHA] Alguns testes falharam. Verifique a saida acima.
    goto :error
)

echo.
echo =================================================================
echo  SUCESSO! Todos os testes passaram.
echo  Relatorio de cobertura XML gerado em: coverage.xml
echo =================================================================
goto :end

:error
echo.
echo =================================================================
echo  FALHA NA EXECUCAO DOS TESTES
echo =================================================================
exit /b 1

:end
echo.
pause