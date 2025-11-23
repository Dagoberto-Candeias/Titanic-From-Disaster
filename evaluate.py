import logging
import pickle
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def evaluate_model(
    model_path: str,
    test_csv_path: str,
    feature_cols: List[str],
    output_dir: str = 'output/avaliacao'
) -> None:
    """
    Carrega um modelo treinado, avalia em dados de teste e gera um relatório.

    Args:
        model_path (str): Caminho para o arquivo .pkl do modelo.
        test_csv_path (str): Caminho para o arquivo CSV de teste com rótulos.
        feature_cols (List[str]): Lista de colunas a serem usadas como features.
        output_dir (str): Diretório para salvar o relatório
            (padrão: 'output/avaliacao').

    Returns:
        None: Loga métricas no console e salva relatório em arquivo.

    Raises:
        FileNotFoundError: Se modelo ou CSV não forem encontrados.
        KeyError: Se colunas especificadas não existirem nos dados.
        ValueError: Se modelo não for válido ou dados incompatíveis.

    Example:
        evaluate_model(
            'output/models/best_model_pipeline.pkl',
            'train.csv',
            ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        )
    """
    try:
        # Valida caminhos
        model_file = Path(model_path)
        test_file = Path(test_csv_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        if not test_file.exists():
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {test_csv_path}")

        # Carrega o modelo
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        if not hasattr(model, 'predict'):
            raise ValueError("Modelo carregado não possui método 'predict'.")
        logging.info(f"Modelo carregado com sucesso de {model_path}")

        # Carrega os dados de teste
        test = pd.read_csv(test_csv_path)
        logging.info(f"Dados de teste carregados de {test_csv_path}")

        # Verifica se colunas existem
        missing_cols = set(feature_cols) - set(test.columns)
        if missing_cols:
            raise KeyError(f"Colunas ausentes nos dados: {missing_cols}")
        if 'Survived' not in test.columns:
            raise KeyError("Coluna 'Survived' não encontrada nos dados.")

        # Separa features (X_test) e labels (y_test)
        X_test = test[feature_cols]
        y_test = test['Survived']

        # Trata valores ausentes (NaN) nas features
        initial_shape = X_test.shape[0]
        nan_mask = X_test.isnull().any(axis=1)
        if nan_mask.any():
            logging.warning(
                f"Encontrados {nan_mask.sum()} linhas com valores NaN "
                "nas features. Removendo essas linhas."
            )
            X_test = X_test[~nan_mask]
            y_test = y_test[~nan_mask]
        final_shape = X_test.shape[0]
        if initial_shape != final_shape:
            logging.info(
                f"Dados reduzidos de {initial_shape} para {final_shape} "
                "linhas após remoção de NaN."
            )

        # Valida consistência após remoção
        if len(X_test) != len(y_test):
            raise ValueError("Inconsistência: X_test e y_test têm tamanhos diferentes.")
        if len(X_test) == 0:
            raise ValueError("Nenhum dado restante após remoção de NaN.")

        # Gera previsões com tratamento de erro
        try:
            y_pred = model.predict(X_test)
        except Exception as pred_error:
            logging.error(f"Erro ao gerar previsões: {pred_error}")
            raise ValueError("Falha na predição - verifique compatibilidade dos dados.")

        # Calcula acurácia
        accuracy = accuracy_score(y_test, y_pred)

        # Gera relatório de classificação
        report = classification_report(y_test, y_pred)

        # Loga métricas no console
        logging.info(f"Acurácia: {accuracy:.4f}")
        logging.info(f"Relatório de Classificação:\n{report}")

        # Cria diretório se não existir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Salva relatório em arquivo
        report_file = output_path / 'evaluation_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Acurácia: {accuracy:.4f}\n\n")
            f.write("Relatório de Classificação:\n")
            f.write(report)
        logging.info(f"Relatório salvo em {report_file}")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except KeyError as e:
        logging.error(f"Erro: Coluna não encontrada nos dados - {e}")
    except ValueError as e:
        logging.error(f"Erro de validação: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")


if __name__ == "__main__":
    model_path = 'output/models/best_model_pipeline.pkl'
    test_csv_path = 'train.csv'
    feature_cols = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
    ]
    evaluate_model(model_path, test_csv_path, feature_cols)
