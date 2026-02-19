@echo off
REM Script para ativar ambiente titanic_ml e rodar testes

setlocal enabledelayedexpansion

echo.
echo ========================================
echo TESTE DO AMBIENTE TITANIC_ML
echo ========================================
echo.

echo [1/3] Ativando ambiente...
call conda activate titanic_ml

echo.
echo [2/3] Verificando imports principais...
python -c "import pandas as pd; import numpy as np; import sklearn; import scipy; print(f'✓ pandas {pd.__version__}'); print(f'✓ numpy {np.__version__}'); print(f'✓ sklearn {sklearn.__version__}'); print(f'✓ scipy {scipy.__version__}')" || goto :ERROR

echo.
echo [3/3] Testando pipeline...
python -c "from titanic_pipeline.core.pipeline import TitanicPipeline; print('✓ TitanicPipeline imports OK')" || goto :ERROR

echo.
echo ========================================
echo TODAS AS VERIFICACOES PASSARAM!
echo ========================================
echo.
goto :END

:ERROR
echo.
echo ❌ ERRO DURANTE TESTE
echo Consulte FIX_NUMPY_COMPATIBILITY.md para solucoes
echo.

:END
pause
