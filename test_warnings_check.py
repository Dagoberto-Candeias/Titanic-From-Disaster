"""
Script para detectar warnings de deprecação e compatibilidade.
Sem suprimir warnings globais - deixa eles aparecerem.
"""
import sys
import os
import warnings

# Ativar TODOS os warnings (não suprimir)
warnings.simplefilter("always")

# Adicionar projeto ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar e testar principais módulos
try:
    print("=" * 70)
    print("TESTANDO IMPORTS E DETECTANDO WARNINGS DE DEPRECAÇÃO")
    print("=" * 70)
    print()

    # Test 1: Imports básicos
    print("[1/5] Importando pandas, numpy, sklearn...")
    import pandas as pd
    import numpy as np
    from sklearn import __version__
    print(f"  ✓ sklearn {__version__}")
    print()

    # Test 2: Importar pipeline principal
    print("[2/5] Importando titanic_pipeline...")
    from titanic_pipeline.core.pipeline import TitanicPipeline
    print("  ✓ TitanicPipeline importada")
    print()

    # Test 3: Carregar dados simples
    print("[3/5] Carregando dados...")
    df = pd.DataFrame({
        'PassengerId': [1, 2, 3],
        'Survived': [0, 1, 1],
        'Age': [22.0, 38.0, 26.0]
    })
    print(f"  ✓ DataFrame criado: {df.shape}")
    print()

    # Test 4: Testar conversão de tipos (common deprecation source)
    print("[4/5] Testando operações com dtypes...")
    # Test numpy dtype usage
    try:
        _ = pd.Series([1, 2, 3], dtype=np.int64)
        print("  ✓ np.int64 dtype ainda funciona")
    except Exception as e:
        print(f"  ✗ Erro com np.int64: {e}")
    
    # Test numpy float types
    try:
        _ = pd.Series([1.0, 2.0], dtype=np.float64)
        print("  ✓ np.float64 dtype ainda funciona")
    except Exception as e:
        print(f"  ✗ Erro com np.float64: {e}")
    print()

    # Test 5: Verificar versões
    print("[5/5] Versões das dependências...")
    print(f"  pandas: {pd.__version__}")
    print(f"  numpy: {np.__version__}")
    print(f"  sklearn: {__version__}")
    print()
    
    print("=" * 70)
    print("TESTE COMPLETADO")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
