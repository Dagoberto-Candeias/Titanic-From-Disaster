# =============================================================================
# TITANIC - SCRIPT DE TREINAMENTO
# Autor: Dagoberto Candeias de Moraes (118550)
# Disciplina: ELT579 - Aprendizado de Máquina
# Versão: 6.0 (Pipeline-driven)
# =============================================================================

import logging
import logging.config
import os
import sys
from typing import Dict, Any

# Imports do pipeline Titanic
from titanic_pipeline.core import TitanicPipeline

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "root": {
        "level": logging.INFO,
        "handlers": ["console"],
    },
}

# Adiciona o diretório raiz ao path para permitir importações do pacote
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuração de logging (pode vir de um arquivo de config)
logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main(config_override: Dict[str, Any] = None):
    """
    Executa o pipeline completo de ML para o Titanic usando a classe orquestradora.
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO - TITANIC (v6.0 Pipeline-driven)")
    logger.info("=" * 80)
    try:
        # 1. Inicializa o pipeline com as configurações
        pipeline = TitanicPipeline(config_override=config_override)

        # 2. Define os caminhos dos dados
        train_path = "data/raw/train.csv"
        test_path = "data/raw/test.csv"

        # 3. Executa o pipeline completo
        results = pipeline.run_pipeline(train_path, test_path)

        # 4. Gera e exibe o sumário final
        if results.get("success"):
            summary = pipeline.get_pipeline_summary(results)
            logger.info("\n" + summary)
            logger.info("✅ SUCESSO TOTAL! PIPELINE DE TREINAMENTO CONCLUÍDO!")
            return True
        else:
            logger.error("❌ FALHA NO PIPELINE: %s", results.get("error"))
            return False

    except Exception as e:
        logger.critical("ERRO CRÍTICO NO ORQUESTRADOR: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    # Exemplo de como passar configurações customizadas via linha de comando no futuro
    # import argparse
    # parser = argparse.ArgumentParser()
    # ...
    # main(config_override=vars(args))
    main()
