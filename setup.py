from setuptools import setup, find_packages

setup(
    name="titanic_pipeline",
    version="1.0.0",
    description="Titanic ML Pipeline",
    author="Dagoberto Candeias de Moraes",
    author_email="dagoberto@email.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy<2",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
        "scipy",
        "joblib",
        "imbalanced-learn",
        "pytest",
        "pytest-cov",
        "pytest-mock",
    ],
)
