@echo off
REM =============================================================================
REM Script de Atualização de Ambiente - Titanic ML Project
REM Atualiza o ambiente conda 'titanic_ml' com base no arquivo environment.yml.
REM Adiciona novos pacotes e remove os que não são mais necessários (--prune).
REM =============================================================================

echo ========================================
echo  ATUALIZANDO AMBIENTE CONDA 'titanic_ml'
echo ========================================

REM --- Verificação do Diretório ---
if not exist "environment.yml" (
    echo.
    echo ERRO: O script deve ser executado a partir da raiz do projeto.
    echo Arquivo 'environment.yml' nao encontrado.
    goto :error
)

REM --- Ativação do Ambiente (necessária para que o conda saiba qual ambiente atualizar se o nome não for passado) ---
echo.
echo [PASSO 1/2] Ativando ambiente conda 'titanic_ml' para garantir o contexto...
call conda activate titanic_ml
if %ERRORLEVEL% NEQ 0 (
    echo AVISO: Nao foi possivel ativar o ambiente 'titanic_ml'.
    echo O comando de atualizacao tentara encontra-lo pelo nome.
) else (
    echo Ambiente ativado.
)

REM --- Execução da Atualização ---
echo.
echo [PASSO 2/2] Executando a atualizacao do ambiente...
echo (Isso pode levar alguns minutos)
conda env update --name titanic_ml --file environment.yml --prune

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FALHA] A atualizacao do ambiente falhou. Verifique a saida acima.
    goto :error
)

echo.
echo =================================================================
echo  SUCESSO! O ambiente 'titanic_ml' foi atualizado.
echo =================================================================
goto :end

:error
echo.
echo =================================================================
echo  FALHA NA ATUALIZACAO DO AMBIENTE
echo =================================================================
exit /b 1

:end
echo.
pause