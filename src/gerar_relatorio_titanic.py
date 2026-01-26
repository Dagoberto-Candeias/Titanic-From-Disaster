import subprocess
import sys

def check_dependencies(dependencies):
    missing = []
    for dependency in dependencies:
        try:
            __import__(dependency)
        except ImportError:
            missing.append(dependency)
    if missing:
        raise ImportError(f"As seguintes dependências estão faltando: {', '.join(missing)}")

import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import shutil
from datetime import datetime
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.inspection import permutation_importance

dependencies = ['pandas', 'seaborn', 'matplotlib', 'sklearn', 'scipy']
check_dependencies(dependencies)

# Ajuste de importação para funcionar tanto como módulo quanto script
try:
    # Se executado como parte de um pacote maior
    from .relatorio_utils import RelatorioBuilder
except ImportError:
    # Se executado como script principal
    from relatorio_utils import RelatorioBuilder

# Configuração de Estilo dos Gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# --- Constantes de Caminho ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
GRAFICOS_DIR = os.path.join(OUTPUT_DIR, 'graficos')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
RELATORIOS_DIR = os.path.join(OUTPUT_DIR, 'relatorios')

def carregar_dados(clean=True):
    """Carrega os dados. Se clean=True, realiza limpeza e imputação."""
    try:
        # Busca robusta pelo arquivo (diretório atual ou relativo ao script)
        possible_paths = [
            os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.csv'),
            os.path.join(PROJECT_ROOT, 'train.csv'),
            'data/raw/train.csv', # Fallback
            'train.csv' # Fallback
        ]
        
        found_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if found_path:
            print(f"✅ Arquivo de dados encontrado em: {found_path}")
            df = pd.read_csv(found_path)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("❌ ERRO CRÍTICO: Arquivo 'train.csv' não encontrado.")
        print("   Certifique-se de que o arquivo está na pasta raiz ou em 'data/raw/'.")
        exit(1)

    if clean:
        # Tratamento básico
        # Extração de Título para imputação de idade mais inteligente
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # Preencher idade com a mediana do grupo de título
        df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
        df['Age'] = df['Age'].fillna(df['Age'].median()) # Fallback

        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def gerar_graficos_eda(df, df_raw=None):
    """Gera os gráficos de Análise Exploratória de Dados."""
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    imgs = {}
    
    # Gráfico 1: Sobrevivência por Sexo
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Sex', hue='Survived', palette='viridis')
    plt.title('Sobrevivência por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Contagem')
    plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
    imgs['sexo'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_sexo.png')
    plt.savefig(imgs['sexo'])
    plt.close()

    # Gráfico 2: Sobrevivência por Classe
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Pclass', palette='magma', errorbar=None, legend=False)
    plt.title('Taxa de Sobrevivência por Classe')
    plt.xlabel('Classe (1 = Alta, 3 = Baixa)')
    plt.ylabel('Taxa de Sobrevivência (0 a 1)')
    imgs['classe'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_classe.png')
    plt.savefig(imgs['classe'])
    plt.close()

    # Gráfico 3: Distribuição de Idade
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, element="step", palette='coolwarm')
    plt.title('Distribuição de Idade e Sobrevivência')
    imgs['idade'] = os.path.join(GRAFICOS_DIR, 'distribuicao_idade.png')
    plt.savefig(imgs['idade'])
    plt.close()

    # Gráfico 4: Correlação (Heatmap)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de Calor de Correlação')
    imgs['corr'] = os.path.join(GRAFICOS_DIR, 'correlacao.png')
    plt.savefig(imgs['corr'])
    plt.close()

    # Gráfico 5: Proporção de Sobreviventes (Pizza)
    plt.figure(figsize=(6, 6))
    df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Não Sobreviveu', 'Sobreviveu'], colors=['lightcoral', 'lightgreen'], startangle=90)
    plt.title('Proporção Total de Sobreviventes')
    plt.ylabel('')
    imgs['pie'] = os.path.join(GRAFICOS_DIR, 'proporcao_sobreviventes.png')
    plt.savefig(imgs['pie'])
    plt.close()

    # Gráfico 6: Embarked
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Embarked', hue='Survived', palette='viridis')
    plt.title('Sobrevivência por Porto de Embarque')
    imgs['embarked'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_embarked.png')
    plt.savefig(imgs['embarked'])
    plt.close()

    # Gráfico 7: Fare
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Survived', y='Fare', hue='Survived', palette='Pastel1', legend=False)
    plt.title('Distribuição de Tarifas por Sobrevivência')
    plt.yscale('log')
    imgs['fare'] = os.path.join(GRAFICOS_DIR, 'distribuicao_fare.png')
    plt.savefig(imgs['fare'])
    plt.close()

    # Gráfico 8: Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='FamilySize', y='Survived', hue='FamilySize', palette='Spectral', errorbar=None, legend=False)
    plt.title('Taxa de Sobrevivência por Tamanho da Família')
    imgs['family'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_familia.png')
    plt.savefig(imgs['family'])
    plt.close()

    # Gráfico 9: Chi2
    categorical_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
    chi2_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 == col2:
                chi2_matrix.loc[col1, col2] = 0.0
            else:
                contingency = pd.crosstab(df[col1], df[col2])
                _, p, _, _ = chi2_contingency(contingency)
                chi2_matrix.loc[col1, col2] = p
    plt.figure(figsize=(8, 6))
    sns.heatmap(chi2_matrix, annot=True, cmap='coolwarm_r', fmt=".2e")
    plt.title('P-valores do Teste Qui-Quadrado')
    imgs['chi2'] = os.path.join(GRAFICOS_DIR, 'chi2_heatmap.png')
    plt.savefig(imgs['chi2'])
    plt.close()

    # Gráfico 10: AgeGroup
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 60, 100], labels=['Criança', 'Adulto', 'Idoso'])
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='AgeGroup', y='Survived', hue='AgeGroup', palette='muted', errorbar=None, legend=False)
    plt.title('Taxa de Sobrevivência por Faixa Etária')
    imgs['age_group'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_faixa_etaria.png')
    plt.savefig(imgs['age_group'])
    plt.close()

    # Gráfico 11: NameLength
    df['NameLength'] = df['Name'].apply(len)
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='NameLength', hue='Survived', kde=True, element="step", palette='coolwarm')
    plt.title('Distribuição do Comprimento do Nome')
    imgs['name_length'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_nome_comprimento.png')
    plt.savefig(imgs['name_length'])
    plt.close()

    # Gráfico 12: Title
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='Title', y='Survived', hue='Title', palette='viridis', errorbar=None, legend=False)
    plt.title('Sobrevivência por Título')
    imgs['title'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_titulo.png')
    plt.savefig(imgs['title'])
    plt.close()

    # Gráfico 13: Deck
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='Deck', y='Survived', hue='Deck', palette='magma', errorbar=None, order=sorted(df['Deck'].unique()), legend=False)
    plt.title('Sobrevivência por Deck')
    imgs['deck'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_deck.png')
    plt.savefig(imgs['deck'])
    plt.close()

    # Gráfico 14: Ticket
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: x.split()[0] if not x.isdigit() else 'None')
    top_prefixes = df['TicketPrefix'].value_counts().nlargest(10).index
    df_ticket = df[df['TicketPrefix'].isin(top_prefixes)]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_ticket, x='TicketPrefix', y='Survived', hue='TicketPrefix', palette='cubehelix', errorbar=None, legend=False)
    plt.title('Sobrevivência por Prefixo do Bilhete')
    plt.xticks(rotation=45)
    imgs['ticket'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_ticket.png')
    plt.savefig(imgs['ticket'])
    plt.close()

    # Gráfico 15: IsAlone (Viajando Sozinho)
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='IsAlone', y='Survived', hue='IsAlone', palette='coolwarm', errorbar=None, legend=False)
    plt.title('Sobrevivência: Viajando Sozinho vs Acompanhado')
    plt.xticks([0, 1], ['Acompanhado', 'Sozinho'])
    plt.ylabel('Taxa de Sobrevivência')
    imgs['is_alone'] = os.path.join(GRAFICOS_DIR, 'sobrevivencia_sozinho.png')
    plt.savefig(imgs['is_alone'])
    plt.close()

    # Gráfico 16: Interação Classe x Sexo
    plt.figure(figsize=(8, 5))
    sns.pointplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='deep', errorbar=None)
    plt.title('Sobrevivência por Classe e Sexo')
    plt.ylabel('Taxa de Sobrevivência')
    imgs['class_sex'] = os.path.join(GRAFICOS_DIR, 'interacao_classe_sexo.png')
    plt.savefig(imgs['class_sex'])
    plt.close()

    # Gráfico 17: Interação Título x Classe
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Title', y='Survived', hue='Pclass', palette='viridis', errorbar=None)
    plt.title('Sobrevivência por Título e Classe')
    plt.ylabel('Taxa de Sobrevivência')
    plt.legend(title='Classe')
    imgs['title_pclass'] = os.path.join(GRAFICOS_DIR, 'interacao_titulo_classe.png')
    plt.savefig(imgs['title_pclass'])
    plt.close()

    # Gráfico 18: Interação Embarked x Classe
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Embarked', y='Survived', hue='Pclass', palette='plasma', errorbar=None)
    plt.title('Sobrevivência por Porto de Embarque e Classe')
    plt.ylabel('Taxa de Sobrevivência')
    plt.legend(title='Classe')
    imgs['embarked_pclass'] = os.path.join(GRAFICOS_DIR, 'interacao_embarked_classe.png')
    plt.savefig(imgs['embarked_pclass'])
    plt.close()

    # Gráfico 19: Valores Nulos (Missing Values)
    if df_raw is not None:
        plt.figure(figsize=(10, 6))
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if not missing.empty:
            ax = sns.barplot(x=missing.index, y=missing.values, palette='Reds_r')
            plt.title('Contagem de Valores Ausentes (Antes do Tratamento)')
            plt.ylabel('Quantidade de Nulos')
            # Adicionar rótulos de porcentagem
            for i, v in enumerate(missing.values):
                ax.text(i, v + 5, f"{v} ({v/len(df_raw):.1%})", ha='center', fontweight='bold')
        
        imgs['missing'] = os.path.join(GRAFICOS_DIR, 'valores_ausentes.png')
        plt.savefig(imgs['missing'])
        plt.close()
    
    return imgs

def treinar_modelo_baseline(df):
    """Treina um modelo simples para servir de baseline no relatório."""
    print("Treinando modelo baseline (Random Forest)...")
    
    # Preparação de Features (Simplificada para Baseline)
    # As features já foram criadas em gerar_graficos_eda e estão no dataframe `df`.
    le = LabelEncoder()
    df_ml = df.copy()
    df_ml['Sex'] = le.fit_transform(df_ml['Sex'])
    df_ml['Embarked'] = le.fit_transform(df_ml['Embarked'])
    df_ml['Title'] = le.fit_transform(df_ml['Title'])
    df_ml['Deck'] = le.fit_transform(df_ml['Deck'])
    df_ml['AgeGroup'] = le.fit_transform(df_ml['AgeGroup'])
    
    X = df_ml[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Deck', 'Title', 'Embarked', 'NameLength', 'IsAlone', 'AgeGroup']]
    y = df_ml['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Comparação de Modelos
    print("Comparando modelos (Random Forest vs Logistic Regression vs Gradient Boosting)...")
    models_comp = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    comp_results = {name: cross_validate(m, X_train, y_train, cv=5, scoring='accuracy')['test_score'].mean() for name, m in models_comp.items()}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(comp_results.keys()), y=list(comp_results.values()), hue=list(comp_results.keys()), palette='viridis', legend=False)
    plt.title('Comparação de Acurácia (Validação Cruzada)')
    plt.ylabel('Acurácia Média')
    plt.ylim(0, 1.0)
    img_model_comp = os.path.join(GRAFICOS_DIR, 'baseline_comparacao_modelos.png')
    plt.savefig(img_model_comp)
    plt.close()

    print("Otimizando hiperparâmetros (Grid Search)...")
    # Define um 'grid' de parâmetros para testar
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    # Busca a melhor combinação
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Melhores parâmetros encontrados: {best_params}")
    
    # Validação Cruzada (5 folds) para avaliar robustez
    cv_results = cross_validate(model, X, y, cv=5, scoring='accuracy')
    cv_mean = cv_results['test_score'].mean()
    cv_std = cv_results['test_score'].std()
    
    # Salvar o modelo treinado
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Versionamento automático
    base_name = "baseline_model"
    version = 1
    while os.path.exists(os.path.join(MODELS_DIR, f"{base_name}_v{version}.pkl")):
        version += 1
    
    model_filename = f"{base_name}_v{version}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Modelo Baseline salvo (versão {version}) em: {model_path}")

    # Manter uma cópia como 'baseline_model.pkl' para compatibilidade com scripts de predição
    default_path = os.path.join(MODELS_DIR, 'baseline_model.pkl')
    if os.path.exists(model_path):
        shutil.copy2(model_path, default_path)
        print(f"[INFO] Modelo padrão atualizado em: {default_path}")
    else:
        print(f"[AVISO] Arquivo {model_path} não encontrado, pulando cópia para {default_path}")

    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Métricas adicionais
    prec = report['weighted avg']['precision']
    rec = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    # ROC
    y_probas = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probas)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    img_roc = os.path.join(GRAFICOS_DIR, 'baseline_roc_curve.png')
    plt.savefig(img_roc)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão')
    img_cm = os.path.join(GRAFICOS_DIR, 'baseline_matriz_confusao.png')
    plt.savefig(img_cm)
    plt.close()

    # Visualização de uma Árvore de Decisão
    plt.figure(figsize=(20, 10))
    # Plotando a primeira árvore do Random Forest
    plot_tree(model.estimators_[0], feature_names=X.columns, class_names=['Morreu', 'Sobreviveu'], filled=True, max_depth=3, fontsize=10)
    plt.title('Visualização de uma Árvore de Decisão (Random Forest)')
    img_tree = os.path.join(GRAFICOS_DIR, 'baseline_arvore_decisao.png')
    plt.savefig(img_tree)
    plt.close()
    
    # Importância das Features
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importancia': model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    # Matriz de Correlação das Features do Modelo
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação (Features do Modelo)')
    plt.tight_layout()
    img_corr_model = os.path.join(GRAFICOS_DIR, 'correlacao_features_modelo.png')
    plt.savefig(img_corr_model)
    plt.close()

    # Permutation Importance
    print("Calculando Permutation Importance...")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[perm_sorted_idx], result.importances_mean[perm_sorted_idx])
    plt.xlabel("Permutation Importance (Queda na Acurácia)")
    plt.title("Importância das Features (Permutação)")
    plt.tight_layout()
    img_perm_imp = os.path.join(GRAFICOS_DIR, 'permutation_importance.png')
    plt.savefig(img_perm_imp)
    plt.close()

    return {
        'ml': {'acc': acc, 'cv_mean': cv_mean, 'cv_std': cv_std, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': roc_auc, 'report': report, 'feature_imp': feature_imp, 'best_params': best_params},
        'imgs': {
            'roc': img_roc,
            'cm': img_cm,
            'tree': img_tree,
            'model_comp': img_model_comp,
            'corr_model': img_corr_model,
            'perm_imp': img_perm_imp
        }
    }

def consolidar_relatorios_existentes(builder):
    """
    Analisa a pasta de relatórios, busca conteúdos gerados por outros pipelines (ex: CSV de modelos,
    gráficos SHAP) e os consolida neste relatório mestre. Remove os arquivos originais após a absorção.
    """
    print("\n--- 🔄 Iniciando Consolidação de Relatórios ---")
    
    # Determinar caminhos absolutos baseados na localização do script
    output_rel_dir = RELATORIOS_DIR
    output_graf_dir = GRAFICOS_DIR

    # 1. Consolidar Tabela de Resultados do Pipeline Avançado
    csv_pipeline = os.path.join(output_rel_dir, "resultados_modelos.csv")
    if os.path.exists(csv_pipeline):
        try:
            print(f"   -> Encontrado CSV de pipeline avançado: {csv_pipeline}")
            df_pipe = pd.read_csv(csv_pipeline)
            # Filtrar colunas essenciais para o relatório executivo
            cols_show = [c for c in ['model_name', 'mean_score', 'std_score', 'mean_auc', 'mean_f1'] if c in df_pipe.columns]
            
            if not df_pipe.empty:
                builder.adicionar_titulo("5. Resultados Consolidados (Pipeline Avançado)", 1)
                builder.adicionar_texto(
                    "Além da modelagem local, foram detectados resultados de um pipeline de processamento intensivo. "
                    "A tabela abaixo consolida o desempenho desses modelos (incluindo Ensembles e Boosting)."
                )
                builder.adicionar_tabela(df_pipe[cols_show].round(4), "Performance Comparativa (Pipeline)")
                
                # Excluir arquivo já consolidado
                try:
                    os.remove(csv_pipeline)
                    print(f"   -> Arquivo {csv_pipeline} consolidado e removido.")
                except OSError as e:
                    print(f"   [AVISO] Não foi possível remover {csv_pipeline}: {e}")
        except Exception as e:
            print(f"   [ERRO] Falha ao consolidar CSV: {e}")

    # 2. Incorporar Gráficos SHAP (se existirem na pasta de gráficos)
    shap_files = glob.glob(os.path.join(output_graf_dir, "06_shap_summary_*.png")) + glob.glob(os.path.join(output_graf_dir, "shap", "06_shap_summary_*.png"))
    if shap_files:
        builder.adicionar_titulo("5.1 Interpretabilidade Avançada (SHAP)", 2)
        builder.adicionar_texto(
            "A análise SHAP (SHapley Additive exPlanations) abaixo detalha como cada variável impactou "
            "a decisão dos modelos complexos gerados pelo pipeline."
        )
        for img_path in shap_files:
            # Extrair nome do modelo do arquivo (ex: 06_shap_summary_Random_Forest.png)
            model_name = os.path.basename(img_path).replace("06_shap_summary_", "").replace(".png", "").replace("_", " ")
            builder.adicionar_imagem(img_path, f"Impacto das Variáveis (SHAP): {model_name}")
            print(f"   -> Gráfico SHAP incorporado: {model_name}")

    # 3. Incorporar Relatórios Markdown Externos (ex: relatorio_final.md)
    # Evita ler o próprio relatório que será gerado (Relatorio_Executivo_Titanic.md)
    target_filename = "Relatorio_Executivo_Titanic.md"
    external_md_files = [
        os.path.join(output_rel_dir, "relatorio_final.md"),
        os.path.join(output_rel_dir, "RELATORIO_FINAL_TITANIC.md"),
        os.path.join(output_rel_dir, "parcial_pipeline_avancado.md")
    ]
    
    img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    found_external_md = False
    for md_file in external_md_files:
        # Verifica se existe e se não é o próprio arquivo alvo (case insensitive para Windows)
        if os.path.exists(md_file) and os.path.basename(md_file).lower() != target_filename.lower():
            try:
                print(f"   -> Processando relatório externo: {md_file}")
                with open(md_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if lines:
                    found_external_md = True
                    builder.adicionar_titulo(f"5.2 Análises do Pipeline Complementar", 2)
                    
                    text_buffer = []
                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        
                        # Detectar Imagem Markdown
                        match = img_pattern.match(line)
                        if match:
                            if text_buffer:
                                builder.adicionar_texto("\n".join(text_buffer))
                                text_buffer = []
                            
                            legenda = match.group(1)
                            caminho_img_rel = match.group(2)
                            
                            # Tentar corrigir caminho da imagem se não existir
                            # Assume que a imagem está em output/graficos se o caminho relativo falhar
                            caminho_img = os.path.join(os.path.dirname(md_file), caminho_img_rel)
                            if not os.path.exists(caminho_img):
                                candidate = os.path.join(output_graf_dir, os.path.basename(caminho_img_rel))
                                if os.path.exists(candidate): caminho_img = candidate
                            
                            if os.path.exists(caminho_img):
                                builder.adicionar_imagem(caminho_img, legenda)
                        
                        elif line.startswith('#'):
                            # Tratar títulos do markdown externo como subtítulos
                            if text_buffer:
                                builder.adicionar_texto("\n".join(text_buffer))
                                text_buffer = []
                            builder.adicionar_titulo(line.lstrip('#').strip(), 3)
                        else:
                            text_buffer.append(line)
                    
                    if text_buffer: builder.adicionar_texto("\n".join(text_buffer))
                
                os.remove(md_file)
                print(f"   -> Arquivo {md_file} integrado e removido.")
            except Exception as e:
                print(f"   [ERRO] Falha ao processar {md_file}: {e}")
    
    if not found_external_md:
        print("   [INFO] Nenhum relatório markdown externo encontrado para a seção 5.2.")

def gerar_relatorio_completo():
    print("Iniciando análise de dados...")
    # Carrega dados crus para análise de nulos e dados limpos para o restante
    df_raw = carregar_dados(clean=False)
    df = carregar_dados(clean=True)
    
    imgs_eda = gerar_graficos_eda(df, df_raw)
    # Treina modelo baseline para comparação
    dados_baseline = treinar_modelo_baseline(df)

    builder = RelatorioBuilder("Relatório Executivo: Análise do Titanic")

    # --- INTRODUÇÃO ---
    builder.adicionar_titulo("1. Introdução e Contexto", 1)
    builder.adicionar_texto(
        "Este relatório apresenta uma análise detalhada sobre o naufrágio do RMS Titanic. "
        "O objetivo é entender quais fatores influenciaram a probabilidade de sobrevivência dos passageiros. "
        "Utilizamos técnicas de Ciência de Dados e Aprendizado de Máquina para extrair padrões dos dados históricos."
    )
    builder.adicionar_texto(
        "**Para o leitor leigo:** Imagine que estamos tentando descobrir se a sorte, o dinheiro ou a idade "
        "foram mais importantes para salvar uma vida naquela noite. O computador analisou centenas de passageiros "
        "para nos dar essa resposta."
    )
    builder.adicionar_texto(
        "**Para o especialista:** O dataset passou por pré-processamento (imputação de nulos na idade pela mediana "
        "e moda para embarque). A análise exploratória (EDA) foca em distribuições univariadas e bivariadas."
    )

    # --- DADOS ---
    builder.adicionar_titulo("2. Visão Geral dos Dados", 1)
    builder.adicionar_texto(
        "Abaixo, uma amostra das primeiras linhas do conjunto de dados utilizado. "
        "Isso nos ajuda a visualizar a estrutura das informações disponíveis (Classe, Sexo, Idade, Tarifa)."
    )
    builder.adicionar_tabela(df.head(), "Amostra dos Dados (Primeiras 5 linhas)")
    
    builder.adicionar_titulo("Estatísticas Descritivas", 2)
    builder.adicionar_texto(
        "A tabela a seguir resume matematicamente os dados numéricos (média, desvio padrão, mínimo, máximo)."
    )
    builder.adicionar_texto(
        "**Tratamento de Dados Faltantes:** Para aumentar a precisão, valores ausentes na idade foram preenchidos "
        "usando a mediana da idade correspondente ao título do passageiro (ex: 'Master' usa mediana de crianças)."
    )
    # Simplificando a tabela de stats para caber melhor no PDF/DOCX
    stats_simple = df.describe().loc[['mean', 'min', 'max', 'std']].round(2).reset_index()
    builder.adicionar_tabela(stats_simple, "Resumo Estatístico")

    # --- ANÁLISE DE DADOS FALTANTES ---
    if 'missing' in imgs_eda:
        builder.adicionar_titulo("2.1 Análise de Dados Faltantes", 2)
        builder.adicionar_texto(
            "Antes de prosseguir com a modelagem, é crucial identificar dados ausentes. "
            "O gráfico abaixo mostra quais variáveis possuíam valores nulos no conjunto original."
        )
        builder.adicionar_imagem(imgs_eda['missing'], "Distribuição de Valores Ausentes")
        builder.adicionar_texto("**Tratamento:** A variável 'Cabin' possui muitos nulos e foi simplificada para 'Deck'. 'Age' foi imputada baseada no Título social, e 'Embarked' pela moda.")

    # --- ANÁLISE VISUAL ---
    builder.adicionar_titulo("3. Análise Exploratória Visual", 1)
    
    # Sexo
    builder.adicionar_titulo("3.1 Influência do Gênero", 2)
    builder.adicionar_texto(
        "Historicamente, a regra 'mulheres e crianças primeiro' foi aplicada. "
        "O gráfico abaixo confirma se os dados refletem essa regra."
    )
    builder.adicionar_imagem(imgs_eda['sexo'], "Comparação de Sobrevivência entre Homens e Mulheres")
    builder.adicionar_texto(
        "**Interpretação:** Observa-se uma taxa de sobrevivência significativamente maior para o sexo feminino. "
        "Isso indica uma forte correlação entre gênero e sobrevivência."
    )

    # Classe
    builder.adicionar_titulo("3.2 Influência da Classe Social (Poder Econômico)", 2)
    builder.adicionar_texto(
        "A classe do passageiro (1ª, 2ª ou 3ª) é um indicativo socioeconômico. "
        "Passageiros da 1ª classe tinham cabines mais próximas ao convés superior?"
    )
    builder.adicionar_imagem(imgs_eda['classe'], "Taxa de Sobrevivência por Classe")
    builder.adicionar_texto(
        "**Interpretação:** Passageiros da 1ª classe tiveram as maiores chances de sobrevivência, "
        "enquanto a 3ª classe sofreu as maiores perdas proporcionais."
    )

    # Idade
    builder.adicionar_titulo("3.3 Distribuição de Idade", 2)
    builder.adicionar_texto(
        "Analisamos como a idade impactou as chances. Crianças foram salvas? Idosos tiveram prioridade?"
    )
    builder.adicionar_imagem(imgs_eda['idade'], "Histograma de Idade: Sobreviventes vs Não Sobreviventes")
    builder.adicionar_texto(
        "**Interpretação:** Nota-se um pico de sobrevivência em crianças pequenas (0-5 anos), "
        "validando a prioridade dada aos infantes."
    )

    # Correlação
    builder.adicionar_titulo("3.4 Análise de Correlação (Heatmap)", 2)
    builder.adicionar_texto(
        "O mapa de calor abaixo exibe a correlação entre as variáveis numéricas. "
        "Correlação mede a relação estatística entre duas variáveis."
    )
    builder.adicionar_imagem(imgs_eda['corr'], "Matriz de Correlação das Variáveis")
    builder.adicionar_texto(
        "**Interpretação:**\n"
        "- **Cores Quentes (Vermelho):** Correlação positiva (ambas sobem juntas).\n"
        "- **Cores Frias (Azul):** Correlação negativa (uma sobe, a outra desce).\n"
        "- **Destaque:** Existe uma correlação negativa relevante entre Classe (Pclass) e Tarifa (Fare)."
    )

    # Adicionando seções faltantes da EDA
    builder.adicionar_titulo("3.5 Sobrevivência por Porto de Embarque", 2)
    builder.adicionar_imagem(imgs_eda['embarked'], "Sobrevivência por Porto")

    builder.adicionar_titulo("3.6 Distribuição de Tarifas", 2)
    builder.adicionar_imagem(imgs_eda['fare'], "Distribuição de Tarifas")
    builder.adicionar_texto("**Interpretação:** Tarifas mais altas estão associadas a maior sobrevivência (correlacionado com a 1ª classe).")

    builder.adicionar_titulo("3.7 Tamanho da Família", 2)
    builder.adicionar_imagem(imgs_eda['family'], "Sobrevivência por Tamanho da Família")
    builder.adicionar_texto("**Interpretação:** Famílias pequenas (2-4 pessoas) tiveram melhores chances do que viajantes solitários ou famílias muito grandes.")

    builder.adicionar_titulo("3.8 Faixa Etária (AgeGroup)", 2)
    builder.adicionar_imagem(imgs_eda['age_group'], "Sobrevivência por Faixa Etária")
    builder.adicionar_texto("**Interpretação:** Crianças tiveram prioridade clara. Idosos tiveram a menor taxa de sobrevivência.")

    builder.adicionar_titulo("3.9 Comprimento do Nome", 2)
    builder.adicionar_imagem(imgs_eda['name_length'], "Sobrevivência vs Comprimento do Nome")
    builder.adicionar_texto("**Interpretação:** Nomes mais longos frequentemente indicam status social mais alto (títulos) ou mulheres casadas (nomes formais), correlacionando-se positivamente com a sobrevivência.")

    builder.adicionar_titulo("3.10 Títulos (Extraídos do Nome)", 2)
    builder.adicionar_imagem(imgs_eda['title'], "Sobrevivência por Título")
    builder.adicionar_texto("**Interpretação:** Mulheres ('Mrs', 'Miss') e Mestres ('Master' - meninos) têm alta sobrevivência. Homens adultos ('Mr') têm a menor.")

    builder.adicionar_titulo("3.11 Deck (Cabine)", 2)
    builder.adicionar_imagem(imgs_eda['deck'], "Sobrevivência por Deck")
    builder.adicionar_texto("**Interpretação:** Decks mais altos (B, C, D, E) geralmente têm taxas de sobrevivência maiores. 'U' representa desconhecido.")

    builder.adicionar_titulo("3.12 Prefixo do Bilhete", 2)
    builder.adicionar_imagem(imgs_eda['ticket'], "Sobrevivência por Prefixo de Bilhete")

    builder.adicionar_titulo("3.13 Viajando Sozinho (IsAlone)", 2)
    builder.adicionar_imagem(imgs_eda['is_alone'], "Sobrevivência: Sozinho vs Acompanhado")
    builder.adicionar_texto("**Interpretação:** Viajar acompanhado aumentou as chances de sobrevivência em relação a viajar sozinho.")

    builder.adicionar_titulo("3.14 Interação Classe x Sexo", 2)
    builder.adicionar_imagem(imgs_eda['class_sex'], "Interação Classe e Sexo")
    builder.adicionar_texto("**Interpretação:** Mulheres da 1ª e 2ª classe sobreviveram quase todas. A maior tragédia foi entre homens da 2ª e 3ª classe.")

    builder.adicionar_titulo("3.15 Correlação Categórica (Qui-Quadrado)", 2)
    builder.adicionar_imagem(imgs_eda['chi2'], "Matriz de Correlação das Variáveis Categóricas")
    builder.adicionar_texto(
        "**Interpretação:** Os valores p (p-values) indicam se existe relação estatística significativa entre as variáveis categóricas."
    )

    builder.adicionar_titulo("3.16 Interação Título x Classe", 2)
    builder.adicionar_imagem(imgs_eda['title_pclass'], "Sobrevivência por Título e Classe")
    builder.adicionar_texto(
        "**Interpretação:** Analisando conjuntamente o Título e a Classe, observamos que a classe social impacta a sobrevivência mesmo dentro dos grupos de títulos. "
        "Mulheres ('Mrs', 'Miss') de classes superiores tendem a ter taxas de sobrevivência quase totais, enquanto homens ('Mr') têm baixa sobrevivência independentemente da classe, embora a 1ª classe ainda ofereça uma leve vantagem."
    )

    builder.adicionar_titulo("3.17 Interação Porto de Embarque x Classe", 2)
    builder.adicionar_imagem(imgs_eda['embarked_pclass'], "Sobrevivência por Porto de Embarque e Classe")
    builder.adicionar_texto(
        "**Interpretação:** Esta análise mostra como a classe social interage com o porto de embarque. "
        "Passageiros da 1ª classe que embarcaram em Cherbourg (C) tiveram uma taxa de sobrevivência notavelmente alta, "
        "sugerindo que este porto pode ter tido uma maior proporção de passageiros abastados que tiveram acesso prioritário aos botes salva-vidas."
    )

    # --- MACHINE LEARNING ---
    builder.adicionar_titulo("4. Modelagem de Referência (Baseline)", 1)
    
    builder.adicionar_titulo("4.1 Comparação de Algoritmos", 2)
    builder.adicionar_texto(
        "Para estabelecer uma linha de base, testamos três algoritmos simples: Random Forest, Regressão Logística e Gradient Boosting."
    )
    builder.adicionar_imagem(dados_baseline['imgs']['model_comp'], "Comparação de Acurácia entre Modelos Baseline")

    builder.adicionar_titulo("4.2 Otimização do Random Forest", 2)
    builder.adicionar_texto(
        "O melhor modelo baseline foi um **Random Forest** (Floresta Aleatória). "
        "Para maximizar a performance, incluímos novas variáveis derivadas (como Título e Tamanho da Família) "
        "e realizamos uma otimização automática de hiperparâmetros (Grid Search)."
    )
    
    acc_percent = dados_baseline['ml']['acc'] * 100
    builder.adicionar_texto(
        f"**Acurácia no Teste:** {acc_percent:.2f}%\n"
        f"**Validação Cruzada (5 folds):** {dados_baseline['ml']['cv_mean']*100:.2f}% (±{dados_baseline['ml']['cv_std']*100:.2f}%)\n"
        "A validação cruzada confirma a robustez do modelo ao testá-lo em diferentes subconjuntos dos dados."
    )
    
    builder.adicionar_texto(
        f"**Melhores Parâmetros Encontrados:**\n"
        f"- Árvores (n_estimators): {dados_baseline['ml']['best_params']['n_estimators']}\n"
        f"- Profundidade Máxima (max_depth): {dados_baseline['ml']['best_params']['max_depth']}\n"
        f"- Mínimo para Divisão (min_samples_split): {dados_baseline['ml']['best_params']['min_samples_split']}\n"
        f"- Mínimo por Folha (min_samples_leaf): {dados_baseline['ml']['best_params']['min_samples_leaf']}\n"
        f"- Máximo de Features (max_features): {dados_baseline['ml']['best_params']['max_features']}"
    )

    builder.adicionar_titulo("4.3 O que foi mais importante para o modelo?", 2)
    builder.adicionar_texto(
        "O algoritmo nos diz quais características pesaram mais na decisão de classificar alguém como sobrevivente."
    )
    builder.adicionar_tabela(dados_baseline['ml']['feature_imp'].round(4), "Importância das Variáveis (Feature Importance)")
    
    builder.adicionar_texto(
        "**Análise Técnica:** A 'Feature Importance' derivada do Random Forest (baseada na impureza de Gini) "
        "mostra que Sexo, Idade e Tarifa são preditores cruciais. Isso corrobora a análise visual feita anteriormente."
    )

    builder.adicionar_titulo("4.3.1 Correlação entre Features Selecionadas", 3)
    builder.adicionar_texto(
        "A matriz abaixo mostra como as variáveis utilizadas pelo modelo se relacionam entre si. "
        "Alta correlação entre variáveis independentes (multicolinearidade) pode afetar a interpretação da importância das features, "
        "embora modelos de árvore como Random Forest sejam geralmente robustos a isso."
    )
    builder.adicionar_imagem(dados_baseline['imgs']['corr_model'], "Correlação das Features do Modelo")

    builder.adicionar_titulo("4.3.2 Importância das Features (Permutação)", 3)
    builder.adicionar_texto(
        "A importância por permutação avalia o impacto de cada variável embaralhando seus valores e medindo a queda na acurácia do modelo. "
        "Isso ajuda a identificar quais features são realmente essenciais para a predição, evitando viéses da importância baseada em impureza (Gini)."
    )
    builder.adicionar_imagem(dados_baseline['imgs']['perm_imp'], "Importância das Features via Permutação")

    builder.adicionar_titulo("4.4 Visualização da Árvore de Decisão", 2)
    builder.adicionar_texto(
        "O Random Forest é composto por várias árvores de decisão. Abaixo, visualizamos uma dessas árvores (limitada a profundidade 3 para legibilidade) "
        "para entender como o modelo toma decisões baseadas nas features."
    )
    builder.adicionar_imagem(dados_baseline['imgs']['tree'], "Exemplo de Árvore de Decisão do Modelo")

    builder.adicionar_titulo("4.5 Avaliação de Desempenho Detalhada", 2)
    
    builder.adicionar_titulo("Matriz de Confusão", 3)
    builder.adicionar_imagem(dados_baseline['imgs']['cm'], "Matriz de Confusão")
    builder.adicionar_texto("Mostra onde o modelo acertou e onde errou (Falsos Positivos vs Falsos Negativos).")

    builder.adicionar_titulo("Curva ROC", 3)
    builder.adicionar_imagem(dados_baseline['imgs']['roc'], "Curva ROC")
    builder.adicionar_texto(f"AUC (Área sob a curva): {dados_baseline['ml']['auc']:.2f}. Quanto mais próximo de 1.0, melhor o modelo distingue entre sobreviventes e não sobreviventes.")

    # --- CONSOLIDAÇÃO ---
    consolidar_relatorios_existentes(builder)

    # --- RECOMENDAÇÕES FUTURAS ---
    builder.adicionar_titulo("6. Recomendações Futuras", 1)
    builder.adicionar_texto(
        "Com base na análise realizada, sugerem-se os seguintes passos para aprimorar o modelo e a compreensão dos dados:"
    )
    builder.adicionar_texto(
        "1. **Engenharia de Atributos Avançada:** Investigar a extração de informações mais detalhadas da variável 'Cabin' e 'Ticket', "
        "bem como agrupar sobrenomes para identificar taxas de sobrevivência familiar."
    )
    builder.adicionar_texto(
        "2. **Tratamento de Dados Faltantes:** Experimentar técnicas de imputação múltipla (MICE) ou baseada em modelos (KNN) para a idade."
    )

    # --- CONCLUSÃO ---
    builder.adicionar_titulo("7. Conclusão", 1)
    builder.adicionar_texto(
        "A análise dos dados do Titanic revela que a sobrevivência não foi aleatória. "
        "Protocolos de emergência e status social desempenharam papéis fundamentais."
    )
    builder.adicionar_texto(
        "**Pontos Chave:**\n"
        "1. **Mulheres** tiveram prioridade absoluta.\n"
        "2. **Crianças** pequenas foram protegidas.\n"
        "3. **Classe Social** importou: dinheiro comprou segurança (ou melhor acesso aos botes).\n"
    )
    builder.adicionar_texto(
        "Este relatório demonstra como dados históricos podem ser transformados em conhecimento "
        "acionável e compreensível tanto para o público geral quanto para a academia."
    )
    
    data_atual = datetime.now().strftime("%d/%m/%Y")
    builder.adicionar_texto(f"Relatório gerado automaticamente em: {data_atual}")
    builder.adicionar_texto("-" * 50)
    builder.adicionar_texto("O código fonte completo deste projeto está disponível no GitHub: https://github.com/dagoberto-moraes/titanic-ml-pipeline")

    # --- SALVAR ARQUIVOS ---
    print("Gerando arquivos finais...")
    
    # Determinar caminhos absolutos para garantir consistência
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(RELATORIOS_DIR, exist_ok=True)
    
    # Limpeza preventiva de relatórios antigos conflitantes (ex: gerados pelo pipeline em caixa alta)
    old_md_files = [
        os.path.join(RELATORIOS_DIR, "Relatorio_Executivo_Titanic.md"),
        os.path.join(PROJECT_ROOT, "Relatorio_Executivo_Titanic.md") # Legado
    ]
    for old_md in old_md_files:
        if os.path.exists(old_md):
            try:
                os.remove(old_md)
                print(f"   -> Relatório antigo/duplicado ({old_md}) removido.")
            except: pass

    caminho_pdf = os.path.join(RELATORIOS_DIR, "Relatorio_Executivo_Titanic.pdf")
    builder.salvar_md(os.path.join(RELATORIOS_DIR, "Relatorio_Executivo_Titanic.md"))
    builder.salvar_docx(os.path.join(RELATORIOS_DIR, "Relatorio_Executivo_Titanic.docx"))
    builder.salvar_pdf(caminho_pdf)
    print("Processo concluído com sucesso!")
    print("Dica: Execute 'python src/ler_relatorio_gerado.py' para visualizar o conteúdo do relatório no terminal.")

if __name__ == "__main__":
    gerar_relatorio_completo()