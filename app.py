import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import numpy as np

# --- CONFIGURAÇÃO INICIAL DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Análise de Churn de Clientes")
st.title('Dashboard de Análise e Previsão de Churn')
st.markdown("---")

# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
@st.cache_data
def load_and_preprocess_data():
    with st.spinner('Carregando e processando os dados...'):
        df = pd.read_csv('./telco_customer_churn.csv')
        df['TotalCharges'] = df['TotalCharges'].replace(' ', pd.NA)
        df.dropna(subset=['TotalCharges'], inplace=True)
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        df['LTV'] = df['MonthlyCharges'] * df['tenure']
        
        df_original = df.copy() 
        df = df.drop(columns='customerID')
        colunas_categoricas = df.select_dtypes(include=['object']).columns
        df_pronto = df.drop(columns=colunas_categoricas)
        
        df_dummies = pd.get_dummies(df[colunas_categoricas], drop_first=True)
        df_final = pd.concat([df_pronto, df_dummies], axis=1)
    return df_final, df_original

df_final, df_original = load_and_preprocess_data()


# 2. TREINANDO O MODELO DE MACHINE LEARNING
@st.cache_resource
def train_model(data):
    with st.spinner('Treinando o modelo de Machine Learning...'):
        X = data.drop(columns='Churn')
        y = data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(X_train, y_train)
        
        y_pred = modelo.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        relatorio = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
    return acuracia, relatorio, modelo, X_test, y_test, cm

acuracia, relatorio, modelo, X_test, y_test, cm = train_model(df_final)


# 3. APRESENTANDO AS MÉTRICAS CHAVE
st.header("1. Métricas Chave do Modelo")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Acurácia", f"{acuracia:.2%}")
with col2:
    st.metric("Precisão (Churn)", f"{relatorio['1']['precision']:.2%}")
with col3:
    st.metric("Revocação (Churn)", f"{relatorio['1']['recall']:.2%}")
st.markdown("---")


# 4. ANÁLISE DETALHADA E VISUALIZAÇÕES
st.header("2. Análise Detalhada")

# Filtro Interativo
st.subheader("Filtre o Dataset")
contrato_selecionado = st.selectbox(
    'Escolha o tipo de contrato:',
    ('Todos', 'Month-to-month', 'One year', 'Two year')
)

df_filtrado = df_original.copy()
if contrato_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Contract'] == contrato_selecionado]

col_grafico1, col_grafico2 = st.columns(2)

with col_grafico1:
    st.subheader("Distribuição de Clientes por Churn")
    fig_churn, ax_churn = plt.subplots()
    sns.countplot(x='Churn', data=df_filtrado, ax=ax_churn)
    ax_churn.set_title(f'Clientes por Churn ({contrato_selecionado})')
    st.pyplot(fig_churn)

with col_grafico2:
    st.subheader("Churn por Tipo de Serviço")
    fig_servicos, ax_servicos = plt.subplots()
    servicos = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
    df_servicos = df_filtrado[df_filtrado['Churn']==1][servicos]
    df_servicos_counts = df_servicos.apply(pd.Series.value_counts)
    if 'Yes' in df_servicos_counts.index:
        df_servicos_counts.loc['Yes'].plot(kind='bar', ax=ax_servicos)
        ax_servicos.set_title('Serviços Mais Presentes em Clientes que Abandonaram')
        st.pyplot(fig_servicos)
    else:
        st.write("Não há clientes com churn e esses serviços neste filtro.")
st.markdown("---")

# Visualização de LTV e Churn
st.subheader("Relação entre LTV e Churn")
fig_ltv, ax_ltv = plt.subplots(figsize=(10, 6))
sns.kdeplot(df_original[df_original['Churn'] == 0]['LTV'], label='Não Churn (0)', fill=True, ax=ax_ltv)
sns.kdeplot(df_original[df_original['Churn'] == 1]['LTV'], label='Churn (1)', fill=True, ax=ax_ltv)
ax_ltv.set_title('Distribuição do Lifetime Value (LTV) por Churn')
ax_ltv.set_xlabel('Lifetime Value (LTV)')
ax_ltv.set_ylabel('Densidade')
ax_ltv.legend()
st.pyplot(fig_ltv)
st.markdown("---")


# 5. INSIGHTS ACIONÁVEIS
st.header("3. Insights Acionáveis")

col_insights_1, col_insights_2 = st.columns(2)
with col_insights_1:
    with st.expander("Top 10 Fatores que Levam ao Churn"):
        st.subheader("Fatores de Previsão do Modelo")
        coeficientes = pd.Series(modelo.coef_[0], index=X_test.columns)
        top_fatores = pd.concat([coeficientes.nlargest(5), coeficientes.nsmallest(5)])
        fig_fatores, ax_fatores = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_fatores.values, y=top_fatores.index, ax=ax_fatores)
        ax_fatores.set_title('Top 10 Fatores Mais Importantes na Previsão de Churn')
        ax_fatores.set_xlabel('Impacto na Probabilidade de Churn')
        st.pyplot(fig_fatores)

    with st.expander("Matriz de Confusão do Modelo"):
        st.subheader("Análise de Erros do Modelo")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=['Não Churn (0)', 'Churn (1)'], yticklabels=['Não Churn (0)', 'Churn (1)'])
        ax_cm.set_title('Matriz de Confusão')
        ax_cm.set_ylabel('Real')
        ax_cm.set_xlabel('Previsão')
        st.pyplot(fig_cm)

with col_insights_2:
    with st.expander("Clientes de Alto Risco e Seu Valor"):
        st.subheader("Priorização por LTV (Valor do Cliente)")
        y_proba = modelo.predict_proba(X_test)[:, 1]
        df_test_com_ids = df_original.loc[X_test.index].copy()
        df_risco = df_test_com_ids.copy()
        df_risco['Probabilidade_Churn'] = y_proba
        df_risco['Churn_Real'] = y_test
        df_risco_sorted = df_risco.sort_values('Probabilidade_Churn', ascending=False)
        st.write("Estes são os 10 clientes com maior probabilidade de churn, priorizados por valor (LTV):")
        st.dataframe(df_risco_sorted[['customerID', 'Contract', 'TotalCharges', 'LTV', 'Probabilidade_Churn']].head(10))

# Adicionando o simulador na barra lateral
st.sidebar.header("Simulador de Risco de Churn")
st.sidebar.markdown("Mude as características do cliente e veja como o risco de churn muda.")

st.sidebar.subheader("Dados do Cliente")
tenure = st.sidebar.slider("Tempo de Contrato (meses)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Encargos Mensais", min_value=10.0, max_value=120.0, value=50.0)
contract_type = st.sidebar.selectbox("Tipo de Contrato", ('Month-to-month', 'One year', 'Two year'))

dados_cliente = pd.DataFrame(np.zeros((1, X_test.shape[1])), columns=X_test.columns)
dados_cliente['tenure'] = tenure
dados_cliente['MonthlyCharges'] = monthly_charges
if contract_type == 'One year':
    dados_cliente['Contract_One year'] = 1
elif contract_type == 'Two year':
    dados_cliente['Contract_Two year'] = 1

probabilidade = modelo.predict_proba(dados_cliente)[:, 1]

st.sidebar.markdown("---")
st.sidebar.subheader("Resultado da Previsão")
st.sidebar.write(f"Probabilidade de Churn: **{probabilidade[0]:.2%}**")

st.info("Esta dashboard transforma dados em uma ferramenta estratégica para tomada de decisões de negócio.")

# --- SEÇÃO DE SEGMENTAÇÃO DE CLIENTES ---
st.markdown("---")
st.header("Análise de Segmentação de Clientes")
with st.expander("Explore o Perfil de Clientes"):
    st.subheader("Segmentação usando K-Means")
    
    # Seleciona as colunas numéricas que serão usadas para segmentar
    colunas_segmentacao = ['tenure', 'MonthlyCharges', 'TotalCharges', 'LTV']
    
    # Cria uma cópia do DataFrame para a análise de segmentação
    df_segmentacao = df_final.copy()
    
    # O filtro de `LTV > 0` deve ser aplicado na cópia para não afetar o DataFrame original
    df_segmentacao = df_segmentacao[df_segmentacao['LTV'] > 0].copy()
    
    # Escolha o número de clusters (pode ser ajustado)
    n_clusters = st.slider("Escolha o número de clusters (grupos)", min_value=2, max_value=5, value=3)
    
    # Roda o K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_segmentacao['Cluster'] = kmeans.fit_predict(df_segmentacao[colunas_segmentacao])
    
    # Visualização dos Perfis dos Clusters (Gráficos de Barras)
    st.write("Visualização dos Perfis dos Clusters:")
    
    cluster_profile = df_segmentacao.groupby('Cluster').agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean',
        'LTV': 'mean',
        'Churn': 'mean' # Média de Churn (0=Não, 1=Sim) representa a taxa de churn
    }).reset_index()
    
    # Gráfico 1: Média de LTV por Cluster
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Cluster', y='LTV', data=cluster_profile, palette='viridis', ax=ax1)
    ax1.set_title('Média de Lifetime Value (LTV) por Cluster')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('LTV Médio')
    st.pyplot(fig1)
    
    # Gráfico 2: Taxa de Churn por Cluster
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Cluster', y='Churn', data=cluster_profile, palette='magma', ax=ax2)
    ax2.set_title('Taxa de Churn por Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Taxa de Churn Média')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Formata como porcentagem
    st.pyplot(fig2)
    
    # Tabela com as características detalhadas de cada cluster
    st.write("Características detalhadas de cada Cluster:")
    cluster_profile.columns = ['Cluster', 'Média de Tempo de Contrato', 'Média de Encargos Mensais', 'Média de LTV', 'Taxa de Churn']
    st.dataframe(cluster_profile)
    
    st.write("A coluna 'Taxa de Churn' mostra a porcentagem de clientes que abandonaram em cada grupo. Compare as características médias de cada cluster (tempo de contrato, encargos, LTV) com sua respectiva taxa de churn para entender melhor os perfis de risco.")


# --- SEÇÃO DE INSIGHTS FINAIS ---
st.markdown("---")
st.header("Análise Conclusiva")
st.write("Esta dashboard oferece insights de alto impacto que podem ser usados por equipes de negócio para otimizar estratégias de retenção de clientes.")
st.write("Principais pontos de interesse:")
st.markdown("""
* **Priorização Eficiente:** O modelo identifica a probabilidade de churn com base em fatores como tipo de contrato e tempo de serviço, permitindo que a equipe de retenção foque seus esforços nos clientes de alto risco e com maior LTV.
* **Fatores-Chave de Risco:** A análise de coeficientes do modelo revela os fatores que têm o maior impacto (positivo ou negativo) na probabilidade de churn. Isso pode guiar a equipe de produto para saber o que precisa de atenção.
* **Segmentação Estratégica:** A análise de clusters mostra que perfis de clientes específicos têm maior propensão ao churn. Por exemplo, pode-se notar que clientes com contratos mensais e baixo LTV formam um grupo de alto risco, exigindo uma estratégia de retenção diferenciada.
* **Decisões Orientadas por Dados:** A simulação na barra lateral demonstra o poder de usar o modelo para responder a perguntas de negócio em tempo real, como "O cliente que assina um contrato mensal tem maior probabilidade de churn?".
""")