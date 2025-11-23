#!/usr/bin/env python3
"""
Script para listar todas as features engenheiradas do projeto Titanic ML.
Baseado no ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py

Autor: Dagoberto Candeias de Moraes
Data: 2023
"""


def listar_features_engenheiradas():
    """
    Lista todas as features criadas pela classe AdvancedFeatureEngineer.
    """
    print("=== FEATURES ENGENHEIRADAS NO PROJETO TITANIC ML ===\n")

    # Features básicas originais (para referência)
    features_basicas = [
        "PassengerId",
        "Survived",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    ]

    print("1. FEATURES BÁSICAS (Originais do Dataset):")
    for i, feat in enumerate(features_basicas, 1):
        print(f"   {i}. {feat}")
    print(f"   Total: {len(features_basicas)} features\n")

    # Features engenheiradas da classe AdvancedFeatureEngineer
    features_engenheiradas = [
        # Análise de Títulos
        "Title",  # Título extraído do nome (Mr, Miss, etc.)
        "Title_Group",  # Grupo de título (Adult_Male, Young_Female, etc.)
        # Análise de Cabines
        "Deck",  # Deck da cabine (A, B, C, ..., U=desconhecido)
        "DeckPriority",  # Prioridade do deck (1-9, baseado em localização)
        "HasCabin",  # Se tem cabine (0/1)
        # Features Familiares
        "FamilySize",  # Tamanho da família (SibSp + Parch + 1)
        "IsAlone",  # Se está sozinho (FamilySize == 1)
        "HasSiblings",  # Se tem irmãos/cônjuge (SibSp > 0)
        "HasParentsChildren",  # Se tem pais/filhos (Parch > 0)
        # Análise de Tickets
        "TicketPrefix",  # Prefixo do ticket (NUM se numérico, ou prefixo)
        "TicketFreq",  # Frequência do ticket (grupos de viagem)
        # Features de Interação
        "AgeClass",  # Idade * Classe (Pclass)
        "FarePerPerson",  # Tarifa por pessoa (Fare / FamilySize)
        "AgeSex",  # Idade * Sexo (0=male, 1=female)
        "ClassFare",  # Classe * Tarifa
        "WealthIndicator",  # Tarifa por pessoa (Fare / FamilySize)
        # Features Polinomiais
        "Age_squared",  # Idade ao quadrado
        "Fare_squared",  # Tarifa ao quadrado
        "Fare_log",  # Log da tarifa (log1p)
        # Features Demográficas
        "IsChild",  # Se é criança (< 12 anos)
        "IsElderly",  # Se é idoso (> 60 anos)
        "IsYoungAdult",  # Se é jovem adulto (18-25 anos)
        # Features Compostas
        "Female_FirstClass",  # Mulher da 1ª classe
        "Male_ThirdClass",  # Homem da 3ª classe
        "Child_Female",  # Criança do sexo feminino
    ]

    print("2. FEATURES ENGENHEIRADAS (Criadas pelo AdvancedFeatureEngineer):")
    for i, feat in enumerate(features_engenheiradas, 1):
        print(f"   {i}. {feat}")
    print(f"   Total: {len(features_engenheiradas)} features\n")

    # Features de Target Encoding (apenas para treino)
    target_encoded_features = [
        "Title_Group_SurvivalRate",  # Taxa de sobrevivência por grupo de título
        "Pclass_SurvivalRate",  # Taxa de sobrevivência por classe
        "Sex_SurvivalRate",  # Taxa de sobrevivência por sexo
        "Embarked_SurvivalRate",  # Taxa de sobrevivência por porto de embarque
        "Deck_SurvivalRate",  # Taxa de sobrevivência por deck
    ]

    print("3. FEATURES DE TARGET ENCODING (Apenas para Treino):")
    for i, feat in enumerate(target_encoded_features, 1):
        print(f"   {i}. {feat}")
    print(f"   Total: {len(target_encoded_features)} features\n")

    # Features selecionadas para modelagem (exemplo baseado no código)
    features_para_modelagem = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "DeckPriority",
        "HasCabin",
        "TicketFreq",
        "AgeClass",
        "FarePerPerson",
        "AgeSex",
        "Age_squared",
        "Fare_log",
        "WealthIndicator",
        "Title_Group",
        "IsChild",
        "IsElderly",
        "Female_FirstClass",
        "Male_ThirdClass",
        # Target encodings se aplicável
        "Title_Group_SurvivalRate",
        "Pclass_SurvivalRate",
        "Sex_SurvivalRate",
        "Deck_SurvivalRate",
    ]

    print("4. FEATURES SELECIONADAS PARA MODELAGEM (Exemplo):")
    for i, feat in enumerate(features_para_modelagem, 1):
        print(f"   {i}. {feat}")
    print(f"   Total aproximado: {len(features_para_modelagem)} features\n")

    # Resumo
    total_features = (
        len(features_basicas)
        + len(features_engenheiradas)
        + len(target_encoded_features)
    )
    print("=== RESUMO ===")
    print(f"Features básicas: {len(features_basicas)}")
    print(f"Features engenheiradas: {len(features_engenheiradas)}")
    print(f"Features target encoding: {len(target_encoded_features)}")
    print(f"TOTAL DE FEATURES DISPONÍVEIS: {total_features}")
    print(
        "\nNota: O número exato pode variar dependendo da disponibilidade no dataset de teste."
    )


if __name__ == "__main__":
    listar_features_engenheiradas()
