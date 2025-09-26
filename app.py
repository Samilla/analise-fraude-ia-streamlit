import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import os
import io

# Título da página Streamlit
st.set_page_config(page_title="Agente de Análise de Dados para Detecção de Fraudes", layout="wide")

#---
# Carregamento de credenciais da API do Gemini
# ---
# Obtenha sua chave de API do Gemini aqui: https://aistudio.google.com/app/apikey
# 1. Crie um arquivo chamado `.streamlit/secrets.toml` na mesma pasta do seu `app.py`.
# 2. Adicione a seguinte linha ao arquivo:
#    GEMINI_API_KEY = "sua_chave_aqui"
# A chave de API será carregada automaticamente por este código.

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = api_key
    genai.configure(api_key=api_key)
except KeyError:
    st.error("Chave de API do Gemini não encontrada. Por favor, adicione sua chave ao arquivo `.streamlit/secrets.toml`.")
    st.stop()


#---
# Configuração da interface do usuário
#---
st.title("Agente de Análise de Dados com IA para Fraudes de Cartão de Crédito 🤖💳")
st.markdown("Bem-vindo! Este agente de IA é especializado em análises de dados, Big Data e contabilidade. Ele pode responder a perguntas, gerar gráficos e extrair conclusões de arquivos CSV, especialmente para detecção de fraudes.")

# Instruções de uso
st.info("""
    **Instruções:**
    1.  O arquivo `creditcard.csv` já está carregado para demonstração.
    2.  Você pode fazer perguntas sobre os dados, como:
        -   `Qual a média da coluna 'Amount'?`
        -   `Quais são os tipos de dados?`
        -   `Qual a correlação entre as colunas 'Time' e 'Amount'?`
        -   `Mostre um histograma da coluna 'Class'.`
        -   `Quais as conclusões que você obteve a partir da análise dos dados?`
""")


# ---
# Função para criar e carregar o agente de IA
# ---
@st.cache_resource
def load_agent(df):
    """
    Cria e retorna um agente de LangChain que pode interagir com o DataFrame.
    """
    try:
        llm = GoogleGenerativeAI(model="gemini-pro")
        agent = create_csv_agent(
            llm=llm,
            path=df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True
        )
        return agent
    except Exception as e:
        st.error(f"Erro ao inicializar o agente de IA: {e}")
        return None

# Carregar o arquivo padrão para demonstração
@st.cache_data
def load_csv_data():
    """
    Carrega o arquivo creditcard.csv para demonstração.
    """
    # URL do arquivo creditcard.csv
    csv_url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    try:
        # Nota: Como a URL do Kaggle é uma página, não o arquivo direto, usaremos um arquivo local
        # para a demonstração. Por favor, baixe o arquivo creditcard.csv e coloque na mesma pasta.
        df = pd.read_csv("creditcard.csv")
        return df
    except FileNotFoundError:
        st.error("Arquivo `creditcard.csv` não encontrado. Por favor, baixe-o e coloque na mesma pasta do `app.py`.")
        st.stop()

# ---
# Lógica da interface e do chat
# ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "plot":
            st.plotly_chart(message["content"])

# Carregar o DataFrame
df = load_csv_data()
if df is not None:
    agent = load_agent(df)

    if agent:
        # Demonstrar o agente com perguntas predefinidas
        demonstration_questions = [
            "Qual a média da coluna 'Amount'?",
            "Mostre um histograma da coluna 'Class'.",
            "Quais as variáveis que mais se correlacionam com a variável 'Amount'?",
            "Quais as conclusões que você obteve a partir da análise deste conjunto de dados?",
        ]
        
        # Simular as respostas do agente para as perguntas de demonstração
        if "demonstration_done" not in st.session_state:
            st.session_state.demonstration_done = False
        
        if not st.session_state.demonstration_done:
            st.subheader("Demonstração das capacidades do agente")
            for q in demonstration_questions:
                with st.chat_message("user"):
                    st.markdown(q)
                
                # Simular a resposta do agente (isso pode levar um tempo)
                with st.chat_message("assistant"):
                    with st.spinner(f"Processando a pergunta: '{q}'..."):
                        if "hist" in q.lower():
                            response = agent.run(q)
                            # Assumir que a resposta gerará um gráfico
                            if 'plot' in response.lower():
                                fig = px.histogram(df, x="Class", title="Distribuição da Variável 'Class'")
                                st.plotly_chart(fig)
                                st.session_state.messages.append({"role": "assistant", "content": fig, "type": "plot"})
                            else:
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response, "type": "text"})
                        else:
                            try:
                                response = agent.run(q)
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response, "type": "text"})
                            except Exception as e:
                                st.markdown(f"**Agente:** Não foi possível responder a esta pergunta. Erro: {e}")
                                st.session_state.messages.append({"role": "assistant", "content": f"Não foi possível responder a esta pergunta. Erro: {e}", "type": "text"})
            st.session_state.demonstration_done = True
            st.balloons()
            st.success("Demonstração concluída! Agora você pode fazer suas próprias perguntas.")

        # Campo de entrada para o usuário
        if prompt := st.chat_input("Faça sua pergunta sobre os dados:"):
            st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analisando os dados..."):
                    try:
                        response = agent.run(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response, "type": "text"})
                    except Exception as e:
                        st.markdown(f"**Agente:** Não foi possível responder a esta pergunta. Erro: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Não foi possível responder a esta pergunta. Erro: {e}", "type": "text"})
