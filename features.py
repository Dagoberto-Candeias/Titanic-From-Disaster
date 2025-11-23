"""
Módulo de Engenharia de Features para o Projeto Titanic.

Este arquivo contém a classe AdvancedFeatureEngineer, responsável por criar
e transformar as features usadas no pipeline de Machine Learning.

Autor: Dagoberto Candeias de Moraes (baseado no projeto principal)
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Configura um logger para este módulo
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Engenharia de Features Avançada - Melhorias significativas"""

    def __init__(self):
        self.title_mapping = {
            "Mr": "Adult_Male",
            "Miss": "Young_Female",
            "Mrs": "Adult_Female",
            "Master": "Child_Male",
            "Dr": "Professional",
            "Rev": "Professional",
            "Col": "Military",
            "Major": "Military",
            "Mlle": "Young_Female",
            "Countess": "Nobility",
            "Ms": "Adult_Female",
            "Lady": "Nobility",
            "Jonkheer": "Nobility",
            "Don": "Nobility",
            "Dona": "Nobility",
            "Mme": "Adult_Female",
            "Capt": "Military",
            "Sir": "Nobility",
        }
        self.deck_priority = {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6,
            "G": 7,
            "T": 8,
            "U": 9,
        }
        self.survival_rates = {}
        self.kfold_encodings = {}  # Para armazenar encodings K-Fold

    def create_advanced_features(self, df, is_training=True):
        """Cria features avançadas baseadas nas versões profissionalizadas"""
        logger.info("🛠️  CRIANDO FEATURES AVANÇADAS...")
        start_time = datetime.now()

        # 1. Análise de Títulos
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        df["Title_Group"] = df["Title"].map(self.title_mapping).fillna("Other")

        # 2. Análise de Cabines
        df["Deck"] = df["Cabin"].str[0].fillna("U")
        df["DeckPriority"] = df["Deck"].map(self.deck_priority)
        df["HasCabin"] = (~df["Cabin"].isna()).astype(int)

        # 3. Features Familiares Complexas
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["HasSiblings"] = (df["SibSp"] > 0).astype(int)
        df["HasParentsChildren"] = (df["Parch"] > 0).astype(int)

        # 4. Análise de Tickets
        df["TicketPrefix"] = df["Ticket"].apply(
            lambda x: (
                "NUM" if x.isdigit() else x.split()[0] if len(x.split()) > 1 else "NUM"
            )
        )
        df["TicketFreq"] = df.groupby("Ticket")["Ticket"].transform("count")

        # Imputação precoce para Age e Fare antes de features dependentes
        df = self.advanced_missing_imputation(df)

        # 5. Features de Interação (agora com Age/Fare imputados)
        df["AgeClass"] = df["Pclass"] * df["Age"]
        df["FarePerPerson"] = df["Fare"] / (df["FamilySize"] + 1e-8)
        df["AgeSex"] = df["Age"] * df["Sex"].map({"male": 0, "female": 1})
        df["ClassFare"] = df["Pclass"] * df["Fare"]
        df["WealthIndicator"] = df["Fare"] / (df["FamilySize"] + 1e-8)

        # 6. Features Polinomiais
        df["Age_squared"] = df["Age"] ** 2
        df["Fare_squared"] = df["Fare"] ** 2
        df["Fare_log"] = np.log1p(df["Fare"])

        # 7. Features Demográficas
        df["IsChild"] = (df["Age"] < 12).astype(int)
        df["IsElderly"] = (df["Age"] > 60).astype(int)
        df["IsYoungAdult"] = ((df["Age"] >= 18) & (df["Age"] <= 25)).astype(int)

        # 8. Features Compostas
        df["Female_FirstClass"] = (
            (df["Sex"] == "female") & (df["Pclass"] == 1)
        ).astype(int)
        df["Male_ThirdClass"] = ((df["Sex"] == "male") & (df["Pclass"] == 3)).astype(
            int
        )
        df["Child_Female"] = (df["IsChild"] & (df["Sex"] == "female")).astype(int)

        # 8.1. Novas Features de Interação (feat_ prefix)
        df["feat_Embarked_Pclass_interact"] = (
            df["Embarked"].map({"S": 1, "C": 2, "Q": 3}).fillna(1) * df["Pclass"]
        )
        df["feat_FamilySize_Age_ratio"] = df["FamilySize"] / (df["Age"] + 1e-8)
        df["feat_Title_Fare_interact"] = (
            df["Title_Group"]
            .map(
                {
                    "Adult_Male": 1,
                    "Young_Female": 2,
                    "Adult_Female": 3,
                    "Child_Male": 4,
                    "Professional": 5,
                    "Military": 6,
                    "Nobility": 7,
                    "Other": 8,
                }
            )
            .fillna(8)
            * df["Fare_log"]
        )
        logger.info(
            "   ✅ Adicionadas 3 novas features de interação: feat_Embarked_Pclass_interact, feat_FamilySize_Age_ratio, feat_Title_Fare_interact"
        )

        # 9. Target Encoding (apenas para treino)
        if is_training and "Survived" in df.columns:
            target_features = ["Title_Group", "Pclass", "Sex", "Embarked", "Deck"]
            for feature in target_features:
                if feature in df.columns:
                    self.survival_rates[feature] = df.groupby(feature)[
                        "Survived"
                    ].mean()
                    df[f"{feature}_SurvivalRate"] = df[feature].map(
                        self.survival_rates[feature]
                    )

            # 10. K-Fold Target Encoding para features específicas
            kfold_features = [
                "Title_Group",
                "TicketPrefix",
                "Deck",
                "Embarked",
                "Pclass",
            ]
            df = self.apply_kfold_target_encoding(df, kfold_features, "Survived")

        # 11. Adicionar indicadores de valores ausentes
        df = self.add_missing_indicators(df)

        # 12. Criar bins e categorizações (agora com Age imputado)
        df = self.create_bins_and_categories(df)

        # 13. Validar imputação (apenas para treino)
        if is_training and "Survived" in df.columns:
            # Only validate if imputation flags exist
            if any(
                f"{col}_imputed" in df.columns
                for col in ["Age", "Fare", "Embarked", "Cabin"]
            ):
                self.validate_imputation(df, original_df=df.copy())

        elapsed = datetime.now() - start_time
        logger.info(
            f"✅ Criadas {len([col for col in df.columns if col not in ['Name', 'Ticket', 'Cabin']])} features avançadas em {elapsed.total_seconds():.2f}s"
        )
        return df

    def advanced_missing_imputation(self, df):
        """Imputação avançada de valores ausentes com flags de imputação e lógica mais sofisticada"""
        logger.info("🔧 IMPUTAÇÃO AVANÇADA DE VALORES AUSENTES...")
        start_time = datetime.now()

        # Age - imputação condicional aprimorada
        if "Age" in df.columns and "Title_Group" not in df.columns:
            df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
            df["Title_Group"] = df["Title"].map(self.title_mapping).fillna("Other")

        if "Age" in df.columns:
            df["Age_imputed"] = df["Age"].isna().astype(int)
            # Imputação mais granular: Title_Group, Pclass, Sex, Embarked
            age_imputation = df.groupby(["Title_Group", "Pclass", "Sex", "Embarked"])[
                "Age"
            ].median()

            def impute_age(row):
                if pd.isna(row["Age"]):
                    key = (
                        row["Title_Group"],
                        row["Pclass"],
                        row["Sex"],
                        row.get("Embarked", "S"),
                    )
                    if key in age_imputation.index:
                        imputed_value = age_imputation.loc[key]
                    else:
                        # Fallback para grupos menores
                        key_partial = (row["Title_Group"], row["Pclass"])
                        if key_partial in age_imputation.index.get_level_values(0):
                            imputed_value = age_imputation.xs(
                                key_partial, level=[0, 1]
                            ).median()
                        else:
                            imputed_value = df["Age"].median()
                    return imputed_value
                return row["Age"]

            df["Age"] = df.apply(impute_age, axis=1)
            df["Age"] = df["Age"].fillna(df["Age"].median())

        # Fare - imputação aprimorada por classe, embarque e família
        if "Fare" in df.columns:
            df["Fare_imputed"] = df["Fare"].isna().astype(int)
            # Usar Pclass, Embarked, SibSp+Parch+1 para imputação mais precisa
            df_temp = df.copy()
            df_temp["FamilySize_temp"] = df_temp["SibSp"] + df_temp["Parch"] + 1
            fare_imputation = df_temp.groupby(["Pclass", "Embarked", "FamilySize_temp"])[
                "Fare"
            ].median()

            def impute_fare(row):
                if pd.isna(row["Fare"]):
                    family_size = row["SibSp"] + row["Parch"] + 1
                    key = (
                        row["Pclass"],
                        row.get("Embarked", "S"),
                        family_size,
                    )
                    imputed_value = fare_imputation.get(key)
                    if pd.isna(imputed_value):
                        # Fallback para Pclass e Embarked
                        key_partial = (row["Pclass"], row.get("Embarked", "S"))
                        imputed_value = fare_imputation.get(
                            key_partial, df["Fare"].median()
                        )
                    return imputed_value
                return row["Fare"]

            df["Fare"] = df.apply(impute_fare, axis=1)
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())

        # Embarked - manter simples, mas com flag
        if "Embarked" in df.columns:
            df["Embarked_imputed"] = df["Embarked"].isna().astype(int)
            df["Embarked"] = df["Embarked"].fillna("S")

        # Cabin - flag adicional (não imputar valor, apenas flag)
        if "Cabin" in df.columns:
            df["Cabin_imputed"] = df["Cabin"].isna().astype(int)

        elapsed = datetime.now() - start_time
        logger.info(
            f"✅ Imputação aprimorada concluída em {elapsed.total_seconds():.2f}s"
        )
        return df

    def validate_imputation(self, df, original_df=None):
        """Valida a qualidade da imputação e gera relatório"""
        logger.info("🔍 VALIDANDO IMPUTAÇÃO...")
        validation_report = {}

        cols_to_check = ["Age", "Fare", "Embarked", "Cabin"]
        for col in cols_to_check:
            if col in df.columns:
                imputed_count = df.get(f"{col}_imputed", 0).sum()
                total_count = len(df)
                pct_imputed = (imputed_count / total_count) * 100

                validation_report[col] = {
                    "total_missing": int(imputed_count),
                    "pct_missing": float(pct_imputed),
                    "imputation_method": (
                        "conditional_median"
                        if col == "Age"
                        else (
                            "group_median"
                            if col in ["Fare"]
                            else "mode" if col == "Embarked" else "flag_only"
                        )
                    ),
                }

                logger.info(
                    f"   {col}: {imputed_count}/{total_count} ({pct_imputed:.1f}%) valores imputados"
                )

                # Verificar se imputação faz sentido estatisticamente
                if (
                    col in ["Age", "Fare"]
                    and original_df is not None
                    and col in original_df.columns
                ):
                    original_mean = original_df[col].mean()
                    imputed_mean = df[col].mean()
                    diff_pct = abs(imputed_mean - original_mean) / original_mean * 100
                    validation_report[col]["mean_diff_pct"] = diff_pct
                    if diff_pct > 20:
                        logger.warning(
                            f"   ⚠️  Diferença significativa na média de {col}: {diff_pct:.1f}%"
                        )

        # Salvar relatório
        os.makedirs("output/changelog", exist_ok=True)
        with open("output/changelog/imputation_validation.json", "w") as f:
            json.dump(validation_report, f, indent=2)

        # Also save as CSV for easier viewing
        csv_report = []
        for col, data in validation_report.items():
            if col != "timestamp":
                csv_report.append(
                    {
                        "column": col,
                        "total_missing": data.get("total_missing", 0),
                        "pct_missing": data.get("pct_missing", 0),
                        "imputation_method": data.get("imputation_method", "N/A"),
                        "mean_diff_pct": data.get("mean_diff_pct", "N/A"),
                    }
                )
        csv_df = pd.DataFrame(csv_report)
        csv_df.to_csv("output/changelog/imputation_report.csv", index=False)

        logger.info(
            "   ✅ Relatório de validação salvo em output/changelog/imputation_validation.json e imputation_report.csv"
        )
        return validation_report

    def add_missing_indicators(self, df):
        """Adiciona indicadores de valores ausentes para colunas principais."""
        logger.info("📊 ADICIONANDO INDICADORES DE VALORES AUSENTES...")
        missing_cols = ["Age", "Cabin", "Embarked", "Fare"]
        for col in missing_cols:
            if col in df.columns:
                df[f"feat_{col}_missing"] = df[col].isna().astype(int)
                logger.info(f"   ✅ Flag adicionada para {col}")
        return df

    def create_bins_and_categories(self, df):
        """Cria bins e categorizações para Age e Fare."""
        logger.info("📊 CRIANDO BINS E CATEGORIZAÇÕES...")

        # Age bins
        if "Age" in df.columns:
            df["feat_AgeBin"] = pd.cut(
                df["Age"],
                bins=[0, 12, 18, 35, 60, 80],
                labels=["Child", "Teen", "Adult", "Senior", "Elderly"],
            )
            df["feat_AgeCategory_v2"] = pd.cut(
                df["Age"],
                bins=[0, 18, 35, 60, 100],
                labels=["Young", "Adult", "Middle", "Senior"],
            )
            logger.info("   ✅ Bins de Age criados")

        # Fare bins
        if "Fare" in df.columns:
            fare_bins = pd.qcut(
                df["Fare"], q=4, labels=["Low", "Med", "High", "VeryHigh"]
            )
            df["feat_FareBin"] = fare_bins
            df["feat_FareCategory_v2"] = pd.cut(
                df["Fare"],
                bins=[0, 10, 50, 100, 600],
                labels=["Cheap", "Moderate", "Expensive", "Luxury"],
            )
            logger.info("   ✅ Bins de Fare criados")

        return df

    def kfold_target_encode(self, df, col, target, n_splits=5, prior=10, suffix="_te"):
        """
        Aplica K-Fold Target Encoding com smoothing para evitar data leakage.

        Args:
            df (pd.DataFrame): DataFrame com os dados.
            col (str): Nome da coluna categórica para encoding.
            target (str): Nome da coluna target (binária).
            n_splits (int): Número de folds para CV.
            prior (float): Parâmetro de smoothing (Bayesian prior).
            suffix (str): Sufixo para a nova coluna (ex.: '_te').

        Returns:
            pd.Series: Série com os valores encoded.
        """
        logger.info(f"   Aplicando K-Fold Target Encoding para {col}...")

        # Calcular média global como prior
        global_mean = df[target].mean()
        encoded = np.zeros(len(df))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(df, df[target]):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]

            # Calcular médias por categoria no fold de treino
            fold_means = train_fold.groupby(col)[target].agg(["mean", "count"])

            # Aplicar smoothing: (count * mean + prior * global_mean) / (count + prior)
            smoothed_means = (
                fold_means["count"] * fold_means["mean"] + prior * global_mean
            ) / (fold_means["count"] + prior)

            # Mapear para o fold de validação
            encoded[val_idx] = val_fold[col].map(smoothed_means).fillna(global_mean)

        return pd.Series(encoded, index=df.index, name=f"feat_{col}{suffix}")

    def apply_kfold_target_encoding(self, df, target_features, target_col="Survived"):
        """
        Aplica K-Fold Target Encoding para múltiplas features.

        Args:
            df (pd.DataFrame): DataFrame de treino.
            target_features (list): Lista de features categóricas para encoding.
            target_col (str): Nome da coluna target.

        Returns:
            pd.DataFrame: DataFrame com novas colunas encoded.
        """
        logger.info("🔄 APLICANDO K-FOLD TARGET ENCODING...")
        start_time = datetime.now()

        for col in target_features:
            if col in df.columns and target_col in df.columns:
                encoded_series = self.kfold_target_encode(df, col, target_col)
                df[encoded_series.name] = encoded_series
                self.kfold_encodings[col] = df.groupby(col)[
                    target_col
                ].mean()  # Para uso em teste

        elapsed = datetime.now() - start_time
        logger.info(
            f"   ✅ K-Fold Target Encoding aplicado em {elapsed.total_seconds():.2f}s"
        )
        return df

    def apply_kfold_to_test(self, df, target_features):
        """
        Aplica encodings K-Fold aprendidos no treino para dados de teste.

        Args:
            df (pd.DataFrame): DataFrame de teste.
            target_features (list): Lista de features categóricas.

        Returns:
            pd.DataFrame: DataFrame com encodings aplicados.
        """
        for col in target_features:
            if col in df.columns and col in self.kfold_encodings:
                df[f"feat_{col}_te"] = (
                    df[col]
                    .map(self.kfold_encodings[col])
                    .fillna(df[col].map(self.kfold_encodings[col]).mean())
                )
        return df

    def select_features_via_model(
        self, X_train, y_train, feature_names, method="rf_importance", threshold=0.01
    ):
        """
        Seleciona features usando importância de modelo ou outro critério.

        Args:
            X_train: Dados de treino processados
            y_train: Target
            feature_names: Nomes das features
            method: Método de seleção ('rf_importance', 'mutual_info', etc.)
            threshold: Threshold para seleção

        Returns:
            selected_features: Lista de features selecionadas
            selector: Objeto seletor treinado
        """
        logger.info(f"🎯 SELECIONANDO FEATURES VIA {method.upper()}...")

        if method == "rf_importance":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import SelectFromModel

            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = SelectFromModel(rf, threshold=threshold)
            selector.fit(X_train, y_train)

            selected_mask = selector.get_support()
            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            logger.info(
                f"   ✅ Selecionadas {len(selected_features)}/{len(feature_names)} features"
            )

            # Salvar relatório
            os.makedirs("output/relatorios", exist_ok=True)
            feature_importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": selector.estimator_.feature_importances_,
                    "selected": selected_mask,
                }
            ).sort_values("importance", ascending=False)
            feature_importance_df.to_csv(
                "output/relatorios/selected_features.csv", index=False
            )

            return selected_features, selector

        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, SelectKBest

            selector = SelectKBest(mutual_info_classif, k="all")
            selector.fit(X_train, y_train)

            scores = selector.scores_
            selected_mask = scores > threshold
            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            logger.info(
                f"   ✅ Selecionadas {len(selected_features)}/{len(feature_names)} features via mutual info"
            )

            # Salvar relatório
            os.makedirs("output/relatorios", exist_ok=True)
            feature_importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": scores,
                    "selected": selected_mask,
                }
            ).sort_values("importance", ascending=False)
            feature_importance_df.to_csv(
                "output/relatorios/selected_features_mutual_info.csv", index=False
            )

            return selected_features, selector

        else:
            logger.warning(
                f"   Método {method} não implementado, retornando todas as features"
            )
            return feature_names, None
