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
    # Adicionar diretório raiz ao path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    logger.info(f"Projeto root: {project_root}")
    
    # Importar e executar o main do script documentado
    from src.ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio import main
    
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
