import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import plotly.express as px
import tempfile
import os
import logging
from langchain.agents import AgentExecutor

# Configuração de Logs para Debug (Útil no ambiente local)
logging.basicConfig(level=logging.INFO)

# --- Configuração da Página e da API Key ---
st.set_page_config(
    page_title="Multi Agente Analista de CSV Fiscal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verifica a chave de API
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Chave de API do Gemini não encontrada. Configure GEMINI_API_KEY no secrets.toml.")
    st.stop()

# --- Funções de Inicialização ---

@st.cache_resource
def load_llm_and_memory():
    """Carrega o LLM e configura a memória de conversa."""
    try:
        # Inicializa o LLM (Gemini Flash é ideal para esta tarefa)
        llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=st.secrets["GEMINI_API_KEY"],
            temperature=0.1 # Um pouco de criatividade para interpretações fiscais
        )
        
        # Configura a memória do agente
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # --- PROMPT GENERALIZADO COM FOCO FISCAL ---
        analyst_prompt = """
        Você é um **Multi Agente de IA SUPER ESPECIALISTA** em Contabilidade, Big Data, Análise de Dados e Python.
        Sua principal tarefa é realizar a **Análise Exploratória de Dados (EDA)** detalhada para fins fiscais, contábeis ou de detecção de fraudes.

        **Siga estas regras, implementadas conforme sua solicitação:**
        1. **Foco Fiscal/Contábil:** Suas análises devem procurar por anomalias, padrões e tendências relevantes para balanços, DRE, detecção de gastos irregulares, ou análise de transações.
        2. **Ferramentas:** Use a ferramenta CSV Agent para gerar código Python, realizar cálculos estatísticos e extrair informações.
        3. **Gráficos:** **SEMPRE** que o usuário solicitar uma visualização, utilize a biblioteca **Plotly** para gerar o gráfico. Gere o código Plotly e explique a conclusão do gráfico.
        4. **Memória:** Use o histórico de conversa para manter o contexto e gerar conclusões detalhadas ao final da interação.
        5. **Linguagem:** Responda **sempre em Português**, de forma clara e profissional.
        6. **Conclusões Finais:** Use a memória de chat e as análises para gerar uma seção de conclusões detalhadas sobre o conjunto de dados.
        
        Histórico da Conversa: {chat_history}
        Pergunta do Usuário: {input}
        Resposta:
        """

        # Configura a cadeia de conversa para a memória
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=PromptTemplate(template=analyst_prompt, input_variables=["chat_history", "input"])
        )

        return llm, conversation, memory
    
    except Exception as e:
        st.error(f"Erro ao inicializar o LLM: {e}")
        return None, None, None

@st.cache_resource
def create_data_agent(_llm, _csv_path):
    """Cria o agente CSV usando o caminho do arquivo."""
    try:
        # Passamos o caminho do arquivo, não o DataFrame.
        # Adicionamos handle_parsing_errors=True para que o agente tente se recuperar de erros de formatação na resposta.
        return create_csv_agent(
            _llm, 
            _csv_path, 
            verbose=True, # Definido como True para ajudar a debugar no Streamlit Cloud
            agent_type="openai-functions", # Agente mais moderno e robusto para formatação de saída
            allow_dangerous_code=True,
            handle_parsing_errors=True # Permite que o agente tente corrigir a formatação da própria resposta
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o agente de IA: {e}")
        return None

# --- Inicialização ---
llm, conversation_chain, memory = load_llm_and_memory()

# --- Interface e Execução ---

st.title("🛡️ Multi Agente Analista de CSV Fiscal")
st.markdown("Faça o upload de qualquer arquivo CSV (fraude, contábil, fiscal) para começar a análise.")

# 1. Componente de Upload
uploaded_file = st.sidebar.file_uploader(
    "1. Faça o Upload do Arquivo CSV", 
    type="csv"
)

data_agent_executor = None

if uploaded_file is not None and llm:
    # 2. Salva o arquivo temporariamente para que o agente possa acessá-lo
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 3. Cria o agente com o caminho temporário
    # create_data_agent retorna um AgentExecutor
    data_agent_executor = create_data_agent(llm, tmp_file_path)

    # Armazena o agente na sessão
    if 'data_agent' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.data_agent = data_agent_executor
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = [] # Limpa o chat para o novo arquivo
        st.success(f"Arquivo '{uploaded_file.name}' carregado e agente inicializado!")

        # Simula a pergunta inicial (agora com tratamento de erro robusto)
        initial_q1 = f"O arquivo {uploaded_file.name} foi carregado. Descreva os dados: Quais são as colunas, tipos de dados e número de linhas/colunas?"
        st.session_state.messages.append({"role": "user", "content": initial_q1})
        with st.spinner(f"Agente pensando em: {initial_q1[:50]}..."):
            try:
                # O agente executa a análise inicial
                response = st.session_state.data_agent.run(initial_q1)
            except Exception as e:
                # Se falhar no parsing da resposta inicial, exibe o erro e uma mensagem.
                response = f"O agente inicializou, mas encontrou um erro ao tentar a primeira análise. O problema de 'Output Parsing' pode ocorrer. Tente perguntar novamente ou faça uma pergunta mais simples. Detalhes: {e}"
                st.error(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Mostra o cabeçalho do arquivo
        df_preview = pd.read_csv(tmp_file_path, low_memory=False)
        st.subheader(f"Pré-visualização do Arquivo: {uploaded_file.name}")
        st.dataframe(df_preview.head())
        
        # Remove o arquivo temporário
        os.remove(tmp_file_path)


if 'data_agent' in st.session_state and st.session_state.data_agent:
    # --- Exibição do Histórico do Chat ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # --- Entrada do Usuário (Bloco de Chat) ---
    if prompt := st.chat_input("Faça sua pergunta ao Agente Analista (Ex: Calcule a média da coluna 'Valor' e gere um histograma dela):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agente de IA está analisando os dados..."):
                try:
                    # Envia a pergunta para o agente de dados
                    response = st.session_state.data_agent.run(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                except Exception as e:
                    # Este bloco de exceção captura e exibe erros, permitindo que o app continue
                    error_message = f"O Agente encontrou um erro ao processar sua requisição. Por favor, tente reformular a pergunta ou verificar se o arquivo CSV está bem formatado. Detalhes: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Por favor, faça o upload de um arquivo CSV para começar a análise.")
