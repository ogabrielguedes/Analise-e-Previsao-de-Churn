# 📊 Previsão de Churn de Clientes e Análise de Valor (LTV)

### **Sobre o Projeto**
Este projeto de Data Science e Machine Learning visa resolver um problema crítico de negócio: o abandono de clientes (churn). Utilizando Python e uma abordagem completa de ponta a ponta, construí um modelo preditivo capaz de identificar clientes em risco de cancelar um serviço, priorizando aqueles com maior potencial de valor (Lifetime Value).

---

### **Objetivo do Projeto**
O principal objetivo foi desenvolver uma ferramenta estratégica para que empresas possam:
1.  **Prever** quais clientes têm maior probabilidade de churn.
2.  **Identificar** os principais fatores que levam ao abandono.
3.  **Priorizar** os esforços de retenção, focando nos clientes mais valiosos (com maior LTV).

---

### **Metodologia**
A análise seguiu um processo estruturado de ciência de dados:
1.  **Aquisição e Limpeza de Dados:** Dados de churn de clientes de telecomunicações foram importados de um arquivo CSV. A coluna `TotalCharges` foi tratada e a variável alvo `Churn` foi convertida para um formato numérico.
2.  **Engenharia de Atributos:** Uma nova variável, o `Lifetime Value (LTV)`, foi criada para adicionar uma dimensão financeira à análise.
3.  **Modelagem Preditiva:** O modelo de Regressão Logística foi treinado para prever a probabilidade de churn, utilizando variáveis demográficas e de serviço.
4.  **Avaliação do Modelo:** A performance do modelo foi validada com métricas como acurácia, precisão e revocação.
5.  **Análise de Cluster e Segmentação:** O algoritmo de clustering K-Means foi usado para segmentar clientes, revelando perfis de risco distintos.
6.  **Dashboard Interativa:** Uma dashboard completa foi construída com Streamlit para apresentar todos os insights de forma clara e acionável.

---

### **Principais Insights e Resultados**
A análise e o modelo revelaram as seguintes conclusões-chave:
* O modelo de Machine Learning alcançou uma **acurácia de 79%** na previsão de churn.
* Mais importante, o modelo foi capaz de identificar corretamente **53%** de todos os clientes que, de fato, abandonaram o serviço (revocação). Desses, ele acertou a previsão **62%** das vezes (precisão).
* **Fatores de Alto Risco:** O tipo de contrato (`Month-to-month`) e os encargos mensais elevados foram identificados como os principais fatores que aumentam a probabilidade de churn.
* **Segmentação de Clientes:** A análise de cluster revelou perfis distintos de clientes, permitindo que a empresa entenda melhor a taxa de churn em diferentes segmentos e adapte suas estratégias de retenção.

---

### **Tecnologias Utilizadas**
-   **Python:** Linguagem de programação principal.
-   **Pandas:** Manipulação e limpeza de dados.
-   **Scikit-learn:** Construção e avaliação do modelo de Machine Learning (Regressão Logística e K-Means).
-   **Matplotlib & Seaborn:** Visualização de dados.
-   **Streamlit:** Construção da dashboard interativa.

---

### **Como Executar o Projeto**
1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/ogabrielguedes/Analise-e-Previsao-de-Churn]
    cd [Analise-e-Previsao-de-Churn]
    ```
2.  **Baixe o arquivo de dados:** Baixe o `telco_customer_churn.csv` do Kaggle e coloque-o na mesma pasta do projeto.
3.  **Instale as dependências:**
    ```bash
    pip install pandas scikit-learn matplotlib seaborn streamlit
    ```
4.  **Execute a dashboard:**
    ```bash
    streamlit run app.py
    ```
