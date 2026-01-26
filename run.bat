@echo off
echo.
echo =================================================
echo  EXECUTANDO PIPELINE DE ANALISE DO TITANIC
echo =================================================
echo.

echo --- Passo 1: Gerando relatorio executivo ---
python src/gerar_relatorio_titanic.py

echo.
echo --- Passo 2: Exibindo relatorio gerado (Markdown) ---
python src/ler_relatorio_gerado.py

echo.
echo Processo concluido. Pressione qualquer tecla para sair.
pause > nul
