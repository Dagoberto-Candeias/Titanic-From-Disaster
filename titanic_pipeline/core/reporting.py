"""
Reporting manager for Titanic ML Pipeline.
"""

import logging
import datetime
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportingManager:
    """Manages report generation and output."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path("output")
        self.reports_dir = self.output_dir / "relatorios"

    def generate_reports(
        self,
        model_results: Dict[str, Any],
        feature_cols: List[str],
        X_train: Any,
        y_train: Any
    ) -> None:
        """
        Generate all configured reports.

        Args:
            model_results: Dictionary with model training results
            feature_cols: List of feature column names
            X_train: Training features
            y_train: Training labels
        """
        # Create output directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # persist training data on the instance for backward-compatible helpers
        # some report helper methods expect self.X_train / self.y_train
        self.X_train = X_train
        self.y_train = y_train

        try:
            if self.config.get("generate_md", True):
                self._generate_markdown_report(model_results, feature_cols)

            if self.config.get("generate_docx", True):
                # _generate_docx_report does not require training data; keep
                # call-site simple and backward-compatible.
                self._generate_docx_report(model_results, feature_cols)

            if self.config.get("generate_pdf", True):
                # PDF report does not need training data either.
                self._generate_pdf_report(model_results, feature_cols)

            # Generate additional plots if configured
            if self.config.get("include_calibration_plots", True):
                self._generate_calibration_plots(
                    model_results,
                    X_train,
                    y_train,
                )

            if self.config.get("include_feature_importance", True):
                self._generate_feature_importance_plots(
                    model_results,
                    feature_cols,
                )

        except Exception as e:
            logger.error("   ❌ Report generation failed: %s", e)

    def _generate_markdown_report(
        self,
        model_results: Dict[str, Any],
        feature_cols: List[str],
    ) -> None:
        """Generate Markdown report."""
        try:
            report_path = self.reports_dir / "relatorio_final.md"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# Titanic ML Pipeline - Relatório Final\n\n")

                # Summary section
                f.write("## Resumo Executivo\n\n")
                total_models = len(model_results)
                f.write(f"- **Total de Modelos Treinados:** {total_models}\n")
                f.write(f"- **Total de Features:** {len(feature_cols)}\n")
                best_score = self._get_best_score(model_results)
                f.write(f"- **Melhor Acurácia:** {best_score:.4f}\n\n")

                # Model results table
                f.write("## Resultados dos Modelos\n\n")
                header = (
                    "| Modelo | Acurácia Média | Desvio Padrão | "
                    "Melhor Score |\n"
                )
                sep = (
                    "|--------|---------------|---------------|"
                    "--------------|\n"
                )
                f.write(header)
                f.write(sep)

                for model_name, result in sorted(
                    model_results.items(),
                    key=lambda x: x[1].get("mean_score", 0),
                    reverse=True,
                ):
                    mean_score = result.get("mean_score", 0)
                    std_score = result.get("std_score", 0)
                    best_score = max(result.get("cv_scores", [0]))

                    row = (
                        f"| {model_name} | {mean_score:.4f} | "
                        f"{std_score:.4f} | {best_score:.4f} |\n"
                    )
                    f.write(row)

                f.write("\n")

                # Feature list
                f.write("## Features Utilizadas\n\n")
                for i, feature in enumerate(feature_cols, 1):
                    f.write(f"{i}. {feature}\n")
                f.write("\n")

                # Configuration
                f.write("## Configuração Utilizada\n\n")
                f.write("```json\n")
                import json
                config_json = json.dumps(self.config, indent=2, default=str)
                f.write(config_json)
                f.write("\n```\n")

            logger.info("   📝 Markdown report saved: %s", report_path)

        except Exception as e:
            logger.error("   ❌ Markdown report generation failed: %s", e)

    def _generate_docx_report(
        self,
        model_results: Dict[str, Any],
        feature_cols: List[str],
    ) -> None:
        """Generate DOCX report."""
        try:
            from docx import Document

            doc = Document()
            doc.add_heading("Titanic ML Pipeline - Relatório Final", 0)

            # Summary
            doc.add_heading("Resumo Executivo", level=1)
            total_models = len(model_results)
            best_score = self._get_best_score(model_results)
            txt_models = "Total de Modelos Treinados: " + str(total_models)
            txt_features = "Total de Features: " + str(len(feature_cols))
            txt_best = "Melhor Acurácia: " + f"{best_score:.4f}"
            doc.add_paragraph(txt_models)
            doc.add_paragraph(txt_features)
            doc.add_paragraph(txt_best)

            # Model results table
            doc.add_heading("Resultados dos Modelos", level=1)
            table = doc.add_table(rows=1, cols=4)
            table.style = "Table Grid"

            # Header row
            header_cells = table.rows[0].cells
            header_cells[0].text = "Modelo"
            header_cells[1].text = "Acurácia Média"
            header_cells[2].text = "Desvio Padrão"
            header_cells[3].text = "Melhor Score"

            # Data rows
            for model_name, result in sorted(
                model_results.items(),
                key=lambda x: x[1].get("mean_score", 0),
                reverse=True,
            ):
                row_cells = table.add_row().cells
                row_cells[0].text = model_name
                row_cells[1].text = f"{result.get('mean_score', 0):.4f}"
                row_cells[2].text = f"{result.get('std_score', 0):.4f}"
                best_score = max(result.get("cv_scores", [0]))
                row_cells[3].text = f"{best_score:.4f}"

            # Save document
            docx_path = self.reports_dir / "relatorio_final.docx"
            doc.save(docx_path)
            logger.info("   📄 DOCX report saved: %s", docx_path)

        except ImportError:
            logger.warning("   ⚠️  python-docx not available; skipping DOCX")
        except Exception as e:
            logger.error("   ❌ DOCX report generation failed: %s", e)

    def _generate_pdf_report(
        self,
        model_results: Dict[str, Any],
        feature_cols: List[str],
    ) -> None:
        """Generate PDF report."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import (
                SimpleDocTemplate,
                Table,
                TableStyle,
                Paragraph,
                Spacer,
            )
            from reportlab.lib.styles import getSampleStyleSheet

            pdf_path = self.reports_dir / "relatorio_final.pdf"
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_txt = "Titanic ML Pipeline - Relatório Final"
            title = Paragraph(title_txt, styles["Title"])
            story.append(title)
            story.append(Spacer(1, 12))

            # Summary
            best_score = self._get_best_score(model_results)
            summary_data = [
                ["Total de Modelos Treinados:", str(len(model_results))],
                ["Total de Features:", str(len(feature_cols))],
                ["Melhor Acurácia:", f"{best_score:.4f}"],
            ]

            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]))

            story.append(summary_table)
            story.append(Spacer(1, 12))

            # Model results table
            model_data = [
                ["Modelo", "Acurácia Média", "Desvio Padrão", "Melhor Score"]
            ]
            for model_name, result in sorted(
                model_results.items(),
                key=lambda x: x[1].get("mean_score", 0),
                reverse=True,
            ):
                model_data.append(
                    [
                        model_name,
                        f"{result.get('mean_score', 0):.4f}",
                        f"{result.get('std_score', 0):.4f}",
                        f"{max(result.get('cv_scores', [0])):.4f}",
                    ]
                )

            model_table = Table(model_data)
            model_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]))

            story.append(model_table)

            # Build PDF
            doc.build(story)
            logger.info("   📕 PDF report saved: %s", pdf_path)

        except ImportError:
            logger.warning("   ⚠️  reportlab not available; skipping PDF")
        except Exception as e:
            logger.error("   ❌ PDF report generation failed: %s", e)

    def _generate_calibration_plots(
        self,
        model_results: Dict[str, Any],
        X_train: Any,
        y_train: Any,
    ) -> None:
        """Generate calibration plots for models."""
        try:
            from sklearn.calibration import calibration_curve
            import matplotlib.pyplot as plt

            graficos_dir = self.output_dir / "graficos" / "calibration"
            graficos_dir.mkdir(parents=True, exist_ok=True)

            for model_name, result in model_results.items():
                if "trained_model" in result:
                    model = result["trained_model"]

                    # Get predicted probabilities
                    if hasattr(model, "predict_proba"):
                        prob_pos = model.predict_proba(X_train)[:, 1]
                    else:
                        continue

                    # Calculate calibration curve
                    prob_true, prob_pred = calibration_curve(
                        y_train, prob_pos, n_bins=10
                    )

                    # Create plot
                    plt.figure(figsize=(8, 6))
                    plt.plot(
                        prob_pred,
                        prob_true,
                        marker="o",
                        linewidth=1,
                        label=model_name,
                    )
                    plt.plot(
                        [0, 1],
                        [0, 1],
                        linestyle="--",
                        color="gray",
                        label="Perfectly calibrated",
                    )

                    plt.xlabel("Predicted probability")
                    plt.ylabel("True probability")
                    plt.title(f"Calibration Plot - {model_name}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Save plot
                    plot_path = graficos_dir / f"calibration_{model_name}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

            logger.info("   📊 Calibration plots generated")

        except Exception as e:
            logger.error(f"   ❌ Calibration plot generation failed: {e}")

    def _generate_feature_importance_plots(
        self,
        model_results: Dict[str, Any],
        feature_cols: List[str],
    ) -> None:
        """Generate feature importance plots."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            graficos_dir = self.output_dir / "graficos" / "feature_importance"
            graficos_dir.mkdir(parents=True, exist_ok=True)

            for model_name, result in model_results.items():
                if "trained_model" in result:
                    model = result["trained_model"]

                    # Check if model has feature_importances_
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_

                        # Use feature_cols, but if length mismatch, use indices
                        if len(feature_cols) == len(importances):
                            feature_names = feature_cols
                        else:
                            feature_names = [
                                f"feature_{i}" for i in range(len(importances))
                            ]

                        # Sort features by importance
                        indices = np.argsort(importances)[::-1]
                        top_n = min(20, len(importances))

                        plt.figure(figsize=(10, 8))
                        plt.title(f"Feature Importances - {model_name}")
                        plt.barh(
                            range(top_n),
                            importances[indices][:top_n],
                            align="center",
                        )

                        ytick_labels = [
                            feature_names[i]
                            for i in indices[:top_n]
                        ]
                        plt.yticks(range(top_n), ytick_labels)
                        plt.xlabel("Relative Importance")
                        plt.gca().invert_yaxis()
                        plt.grid(True, alpha=0.3)

                        # Save plot
                        fname = f"feature_importance_{model_name}.png"
                        plot_path = graficos_dir / fname
                        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close()

            logger.info("   📊 Feature importance plots generated")

        except Exception as e:
            logger.error("Feature importance generation failed: %s", e)

    def _get_best_score(self, model_results: Dict[str, Any]) -> float:
        """Get the best score from model results."""
        if not model_results:
            return 0.0

        return max(
            result.get("mean_score", 0) for result in model_results.values()
        )


# Standalone functions for backward compatibility

def generate_reports(
    model_results: Dict[str, Any],
    feature_cols: List[str],
    X_train: Any = None,
    y_train: Any = None,
) -> None:
    """Standalone function to generate reports."""
    manager = ReportingManager({})
    manager.generate_reports(model_results, feature_cols, X_train, y_train)


def generate_roc_curves(model_results, X_train, y_train, feature_cols=None):
    """Generate ROC curves for models."""
    try:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        from pathlib import Path

        roc_dir = Path("output/graficos/roc_curves")
        roc_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))

        for model_name, result in model_results.items():
            if (
                "trained_model" in result
                and hasattr(result["trained_model"], "predict_proba")
            ):
                model = result["trained_model"]

            # Determine which columns to pass to predict_proba.
            # Prefer model.feature_names_in_ when available.
            # Otherwise use feature_cols if present in X_train.
            # Fallback: use X_train as provided.
            X_for_pred = X_train
            if hasattr(X_train, "select_dtypes") and feature_cols:
                expected = getattr(model, "feature_names_in_", None)
                if expected is not None:
                    # If we can select the original training columns
                    # from X_train, do so.
                    if set(expected).issubset(set(X_train.columns)):
                        X_for_pred = X_train[list(expected)]
                    else:
                        # Can't reconstruct the original feature set from
                        # X_train; use X_train as provided.
                        X_for_pred = X_train
                else:
                    # No feature-name metadata on the model; use
                    # feature_cols only if present in X_train.
                    if set(feature_cols).issubset(set(X_train.columns)):
                        X_for_pred = X_train[feature_cols]
                    else:
                        X_for_pred = X_train
                y_pred_proba = model.predict_proba(X_for_pred)[:, 1]
                fpr, tpr, _ = roc_curve(y_train, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(roc_dir / "04_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("   📊 ROC curves generated")
    except Exception as e:
        logger.error(f"   ❌ ROC curve generation failed: {e}")


def generate_feature_correlation_heatmap(train, feature_cols):
    """Generate feature correlation heatmap."""
    try:
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pathlib import Path

        corr_dir = Path("output/graficos/correlation")

        # Filtrar apenas colunas numéricas para correlação
        numeric_cols = (
            train[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        if not numeric_cols:
            logger.warning("   ⚠️  No numeric columns for correlation heatmap")
            # Defensive: if an old heatmap exists from earlier runs, remove it
            # to ensure callers/tests see a consistent state (no heatmap).
            old_path = corr_dir / "09_feature_correlation_heatmap.png"
            try:
                if old_path.exists():
                    old_path.unlink()
            except Exception:
                # Non-fatal: log and continue silently
                logger.debug("   ⚠️  Could not remove old correlation heatmap file")

            return

        # Only create output directory when we actually will save a plot
        corr_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "   📊 Using %d numeric features for correlation heatmap",
            len(numeric_cols),
        )

        corr_matrix = train[numeric_cols].corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        out_path = corr_dir / "09_feature_correlation_heatmap.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("   📊 Feature correlation heatmap generated")
    except Exception as e:
        logger.error(
            "   ❌ Feature correlation heatmap generation failed: %s",
            e,
        )


def generate_model_performance_timeline(model_results):
    """Generate model performance timeline."""
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path

        timeline_dir = Path("output/graficos/timeline")
        timeline_dir.mkdir(parents=True, exist_ok=True)

        # Simple timeline plot
        models = list(model_results.keys())
        scores = [
            result.get("mean_score", 0)
            for result in model_results.values()
        ]
        plt.figure(figsize=(12, 6))
        plt.bar(models, scores)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Mean CV Score')
        plt.title('Model Performance Timeline')
        plt.tight_layout()
        plt.savefig(
            timeline_dir / "10_model_performance_timeline.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("   📊 Model performance timeline generated")
    except Exception as e:
        logger.error(f"   ❌ Model performance timeline generation failed: {e}")


def generate_changelog_and_manifest(
    feature_cols,
    model_results,
    script_total_time,
):
    """Generate changelog and manifest."""
    try:
        changelog_dir = Path("output") / "changelog"
        changelog_dir.mkdir(parents=True, exist_ok=True)

        # Convert script_total_time to float for JSON serialization
        if isinstance(script_total_time, datetime.timedelta):
            total_seconds = script_total_time.total_seconds()
        else:
            total_seconds = float(script_total_time)

        # Manifest
        manifest = {
            "total_time_seconds": total_seconds,
            "features_count": len(feature_cols),
            "models_count": len(model_results),
            "best_score": _get_best_score(model_results)
        }
        with open(changelog_dir / "manifest.json", "w") as f:
            import json
            json.dump(manifest, f, indent=2, default=str)

        # Changelog
        with open(changelog_dir / "CHANGELOG.md", "w") as f:
            f.write("# Changelog\n\n")
            f.write(f"- Generated at: {datetime.datetime.now().isoformat()}\n")
            f.write(f"- Total time: {total_seconds:.2f} seconds\n")
            f.write(f"- Features: {len(feature_cols)}\n")
            f.write(f"- Models: {len(model_results)}\n")

        logger.info("   📝 Changelog and manifest generated")
    except Exception as e:
        logger.error(f"   ❌ Changelog and manifest generation failed: {e}")


def save_timing_report(script_total_time, model_results):
    """Save timing report."""
    try:
        from pathlib import Path

        reports_dir = Path("output/relatorios")
        reports_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(script_total_time, datetime.timedelta):
            total_seconds = script_total_time.total_seconds()
        else:
            total_seconds = float(script_total_time)
        timing = {
            "total_time_seconds": total_seconds,
            "models_trained": len(model_results),
            "best_model": max(
                model_results,
                key=lambda x: model_results[x].get(
                    "mean_score",
                    0,
                ),
            ),
        }
        with open(reports_dir / "timing_report.json", "w") as f:
            import json
            json.dump(timing, f, indent=2, default=str)
        logger.info("   ⏱️  Timing report saved")
    except Exception as e:
        logger.error(f"   ❌ Timing report save failed: {e}")


def generate_shap_comparison_plot(top_models, X_train_data, feature_names_out):
    """Generate SHAP comparison plot."""
    try:
        import shap
        import matplotlib.pyplot as plt
        from pathlib import Path

        shap_dir = Path("output/graficos/shap")
        shap_dir.mkdir(parents=True, exist_ok=True)

        # Simplified SHAP comparison
        plt.figure(figsize=(12, 8))
        for name, perf in top_models:
            model = perf.get("trained_model")
            if model and hasattr(model, "predict"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(
                    X_train_data[:100]
                )  # sample
                shap.summary_plot(
                    shap_values,
                    X_train_data[:100],
                    feature_names=feature_names_out,
                    show=False,
                )
                plt.title(f"SHAP Summary - {name}")
                plt.savefig(
                    shap_dir / "08_shap_comparison.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
                break  # Only first for simplicity
        logger.info("   📊 SHAP comparison plot generated")
    except Exception as e:
        logger.error(f"   ❌ SHAP comparison plot generation failed: {e}")


def improved_generate_submission(final_model, test, feature_cols, train):
    """Generate improved submission."""
    try:
        import joblib
        import pandas as pd
        pipeline = joblib.load("output/models/best_model_pipeline.pkl")
        X_test = test[feature_cols]
        predictions = pipeline.predict(X_test)

        submission = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': predictions.astype(int)
        })
        submission.to_csv("output/submission_titanic_final.csv", index=False)
        logger.info("   📤 Improved submission generated")
    except Exception as e:
        logger.error(f"   ❌ Improved submission generation failed: {e}")


def generate_model_calibration_plots(model, X_train, y_train, model_name):
    """Generate calibration plots for a model."""
    try:
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        if hasattr(model, "predict_proba"):
            try:
                prob_pos = model.predict_proba(X_train)[:, 1]
                prob_true, prob_pred = calibration_curve(
                    y_train, prob_pos, n_bins=10
                )
                plt.plot(prob_pred, prob_true, marker="o", label=model_name)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel("Predicted probability")
                plt.ylabel("True probability")
                plt.title(f"Calibration Plot - {model_name}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                out_path = (
                    f"output/graficos/09_model_calibration_{model_name}.png"
                )
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as inner_e:
                logger.warning(
                    f"   ⚠️  Skipping calibration for {model_name}: {inner_e}"
                )
        logger.info(f"   📊 Calibration plot for {model_name} generated")
    except Exception as e:
        logger.error(f"   ❌ Calibration plot for {model_name} failed: {e}")


def log_model_performance_to_csv(
    model_results: Dict[str, Any],
    output_path: str = "output/reports/model_performance.csv"
) -> None:
    """
    Log model performance metrics to a CSV file.

    Args:
        model_results: Dictionary with model results
        output_path: Path to save the CSV file
    """
    import csv
    import os

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare data for CSV
    rows = []
    for model_name, result in model_results.items():
        row = {
            "model_name": model_name,
            "mean_accuracy": result.get("mean_score", None),
            "std_score": result.get("std_score", None),
            "mean_auc": result.get("mean_auc", None),
            "mean_precision": result.get("mean_precision", None),
            "mean_recall": result.get("mean_recall", None),
            "mean_f1": result.get("mean_f1", None),
            "error": result.get("error", None),
            "cv_scores": result.get("cv_scores", []),
        }
        rows.append(row)

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "model_name",
            "mean_accuracy",
            "std_score",
            "mean_auc",
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "error",
            "cv_scores",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Convert cv_scores list to string for CSV
            row["cv_scores"] = str(row["cv_scores"])
            # Ensure None values are written as empty (CSV-friendly)
            for k, v in row.items():
                if v is None:
                    row[k] = ""
            writer.writerow(row)

    logger.info(f"Model performance logged to {output_path}")


def generate_permutation_importance(
    model,
    X_train,
    y_train,
    feature_names,
    n_repeats: int = 5,
    model_name: str = "Model",
) -> None:
    """Generate permutation importance."""
    try:
        from sklearn.inspection import permutation_importance
        import matplotlib.pyplot as plt
        import pandas as pd
        from pathlib import Path

        # Ensure output directory exists
        graficos_dir = Path("output/graficos")
        graficos_dir.mkdir(parents=True, exist_ok=True)

        perm_importance = permutation_importance(
            model,
            X_train,
            y_train,
            n_repeats=n_repeats,
            random_state=42,
        )
        sorted_idx = perm_importance.importances_mean.argsort()

        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(sorted_idx)),
            perm_importance.importances_mean[sorted_idx],
        )
        plt.yticks(
            range(len(sorted_idx)),
            [feature_names[i] for i in sorted_idx],
        )
        plt.xlabel("Permutation Importance")
        plt.title(f"Permutation Importance - {model_name}")
        plt.tight_layout()

        # Use Path for safe file handling
        safe_model_name = model_name.replace(" ", "_").replace("/", "_")
        plot_path = graficos_dir / (
            f"permutation_importance_{safe_model_name}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save as CSV
        features_list = [feature_names[i] for i in sorted_idx]
        mean_list = perm_importance.importances_mean[sorted_idx]
        std_list = perm_importance.importances_std[sorted_idx]

        df = pd.DataFrame(
            {
                "feature": features_list,
                "importance_mean": mean_list,
                "importance_std": std_list,
            }
        )
        csv_path = graficos_dir / (
            f"permutation_importance_{safe_model_name}.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info(
            f"   📊 Permutation importance for {model_name} generated"
        )
    except Exception as e:
        logger.error(
            f"   ❌ Permutation importance for {model_name} failed: {e}"
        )


def _get_best_score(model_results: Dict[str, Any]) -> float:
    """Get the best score from model results."""
    if not model_results:
        return 0.0

    return max(
        result.get("mean_score", 0)
        for result in model_results.values()
    )
