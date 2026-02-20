@echo off
REM =============================================================================
REM Script de FormataÃ§Ã£o de CÃ³digo - Titanic ML Project
REM Usa o 'black' para formatar automaticamente o cÃ³digo Python nas pastas
REM src/, titanic_pipeline/, e tests/.
REM =============================================================================

echo ========================================
echo  INICIANDO FORMATACAO DE CODIGO COM BLACK
echo ========================================

REM --- AtivaÃ§Ã£o do Ambiente ---
echo.
echo [PASSO 1/2] Ativando ambiente conda 'titanic_ml'...
call conda activate titanic_ml
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Nao foi possivel ativar o ambiente 'titanic_ml'.
    echo Execute 'setup_environment.bat' primeiro.
    goto :error
)
echo Ambiente ativado com sucesso.

REM --- ExecuÃ§Ã£o do Black ---
echo.
echo [PASSO 2/2] Executando 'black' para formatar o codigo...
python -m black src/ titanic_pipeline/ tests/
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: A formatacao com 'black' falhou.
    goto :error
)
echo.
echo Formatacao concluida. Verifique o log acima para ver os arquivos alterados.

echo.
echo =================================================================
echo  ðŸŽ‰ PROCESSO DE FORMATACAO FINALIZADO COM SUCESSO!
echo =================================================================
goto :end

:error
echo.
echo =================================================================
echo  âŒ FALHA NA EXECUCAO!
echo  O processo foi interrompido devido a um erro no passo anterior.
echo =================================================================

:end
echo.
pause