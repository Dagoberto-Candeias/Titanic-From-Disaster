#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para executar o pipeline completo do Titanic sem interrupções.
Persiste logs e resultados em arquivos.
"""

import sys
import os
import logging
from datetime import datetime

# Configurar logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"pipeline_full_run_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("INICIANDO PIPELINE COMPLETO DO TITANIC")
logger.info(f"Timestamp: {timestamp}")
logger.info("="*80)

try:
    # Com o projeto instalado em modo editável (`pip install -e .`),
    # o 'src' é adicionado ao PYTHONPATH, e a manipulação de sys.path não é necessária.

    # Importar e executar a função principal do pipeline de geração de relatório.
    from src.gerar_relatorio_titanic import gerar_relatorio_completo as main
    
    logger.info("Chamando função main()...")
    main()

    logger.info("="*80)
    logger.info("PIPELINE COMPLETO FINALIZADO COM SUCESSO!")
    logger.info("="*80)
    
except Exception as e:
    logger.error("="*80)
    logger.error(f"ERRO DURANTE EXECUÇÃO DO PIPELINE: {e}", exc_info=True)
    logger.error("="*80)
    sys.exit(1)
