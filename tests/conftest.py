"""
Arquivo de configuração para testes pytest.
"""

import pytest

# Aqui você pode adicionar fixtures, configurações, etc. que serão
# automaticamente aplicadas a todos os testes.


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configura o logging para os testes."""
    import logging

    logging.basicConfig(level=logging.INFO)
