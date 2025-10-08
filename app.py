# -*- coding: utf-8 -*-
# Agente de Análise de Dados e Detecção de Fraudes com Gemini SDK (Versão Final Estável)
# Implementado a arquitetura de Chamada de Função (Tool Calling) para resolver o Timeout 500.

import streamlit as st
import pandas as pd
import os
import tempfile
import zipfile
import gzip
import io
import json
import plotly.express as px
import plotly.io as pio
import google.generativeai as genai
import traceback
import operator
from typing import TypedDict, Annotated, List, Union

# Importações de LangChain simplificadas (necessárias para Tool Calling)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Configurações Iniciais do Streamlit ---
st.set_page_config(layout="wide", page_title="Multi Agente de Análise Fiscal e de Fraudes (Estável)")

# --- Constantes e Variáveis Globais ---
pio.templates.default = "plotly_white"
MODEL_NAME = "gemini-2.5-flash"
MAX_HISTORY_SIZE = 10 
SAMPLE_ROWS = 25000 # REDUÇÃO CRÍTICA FINAL para 25.000 linhas para garantir estabilidade e resolver o timeout
DEFAULT_TEMP_CSV_PATH = "/tmp/default_data.csv"

# Tenta obter a chave da API do Gemini do secrets.toml (Streamlit Cloud)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_KEY", "")

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada. Configure a chave no .streamlit/secrets.toml ou na variável de ambiente GEMINI_KEY.")

# --- Inicialização Estável do Gemini (LangChain Tool) ---

@st.cache_resource
def get_llm():
    """Inicializa o modelo de chat Gemini com temperatura zero para precisão."""
    if not API_KEY:
        return None
    try:
        # LLM é criado sem o parâmetro obsoleto 'client_options'
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            temperature=0.0
        )
        return llm
    except Exception as e:
        st.error(f"Erro fatal ao configurar o LLM Gemini: {e}")
        return None

# --- Definição das Ferramentas (Tools) ---

class AnalysisInput(TypedDict):
    """Esquema de entrada para as ferramentas de análise."""
    column: Annotated[str, "O nome exato da coluna a ser analisada no DataFrame (df)."]

class PlotInput(AnalysisInput):
    """Esquema de entrada para as ferramentas de plotagem."""
    plot_type: Annotated[str, "O tipo de gráfico a ser gerado (e.g., 'histogram', 'boxplot', 'scatter')."]

@tool
def describe_dataframe() -> str:
    """Retorna o número de linhas, colunas e os tipos de dados de cada coluna (dtypes) do DataFrame (df)."""
    if st.session_state.df_exec is None:
        return "Erro: Nenhum DataFrame carregado."
    
    col_info = st.session_state.df_exec.dtypes.to_markdown()
    info = (
        f"O DataFrame contém {st.session_state.df_exec.shape[0]} linhas e {st.session_state.df_exec.shape[1]} colunas.\n"
        f"Informações das Colunas:\n{col_info}"
    )
    return info

@tool
def calculate_statistics(column: str) -> str:
    """Calcula estatísticas descritivas (média, mediana, desvio padrão, mínimo, máximo, e contagem de nulos) para uma coluna numérica."""
    df = st.session_state.df_exec
    if df is None or column not in df.columns:
        return f"Erro: Coluna '{column}' não encontrada."

    if pd.api.types.is_numeric_dtype(df[column]):
        stats = df[column].agg(['mean', 'median', 'std', 'min', 'max']).to_dict()
        null_count = df[column].isnull().sum()
        stats['null_count'] = null_count
        return json.dumps(stats, indent=2)
    else:
        return f"Erro: A coluna '{column}' não é numérica para cálculo de estatísticas."

@tool
def generate_plotly_plot(column: str, plot_type: str) -> str:
    """Gera o código Python para um gráfico Plotly, imprime o JSON do gráfico e o retorna."""
    df = st.session_state.df_exec
    if df is None or column not in df.columns:
        return f"Erro: Coluna '{column}' não encontrada."
    
    # Lógica de Plotagem (Plotly)
    try:
        if plot_type == 'histogram':
            fig = px.histogram(df, x=column, title=f'Distribuição de {column}')
        elif plot_type == 'boxplot':
            fig = px.box(df, y=column, title=f'Box Plot de {column}')
        elif plot_type == 'scatter' and len(df.columns) >= 2:
            # Exemplo de scatter plot (usa as duas primeiras colunas como eixo)
            col2 = df.columns[1] if df.columns[0] == column and len(df.columns) > 1 else df.columns[0]
            fig = px.scatter(df, x=column, y=col2, title=f'Relação entre {column} e {col2}')
        else:
            return f"Tipo de gráfico '{plot_type}' não suportado ou dados insuficientes."
            
        # Imprime e retorna o JSON do Plotly para ser capturado pelo parser
        plot_json = fig.to_json()
        
        # O agente LangChain retorna isso. O Streamlit precisa capturar.
        return f"PLOTLY_OUTPUT: <PLOTLY_JSON>{plot_json}</PLOTLY_JSON>"
        
    except Exception as e:
        return f"Erro ao gerar gráfico Plotly para a coluna {column}: {e}"

# Lista de todas as ferramentas disponíveis (acesso global)
tools = [describe_dataframe, calculate_statistics, generate_plotly_plot]

# --- Funções de Manipulação de Arquivos ---

@st.cache_data
def unzip_and_read_file(uploaded_file):
    """Descompacta arquivos ZIP ou GZ, lê o conteúdo CSV e retorna o caminho temporário."""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    uploaded_file.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_csv_path = tmp_file.name

    try:
        # Lógica de descompactação
        if file_extension == 'zip':
            with zipfile.ZipFile(uploaded_file, 'r') as zf:
                csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
                if not csv_files:
                    st.error("Nenhum arquivo CSV encontrado dentro do ZIP.")
                    return None, None
                
                with zf.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)
                    df.to_csv(tmp_csv_path, index=False)
                    return tmp_csv_path, df
        
        elif uploaded_file.name.endswith(('.gz', '.gzip')):
            with gzip.open(uploaded_file, 'rt') as gz_file:
                df = pd.read_csv(gz_file)
                df.to_csv(tmp_csv_path, index=False)
                return tmp_csv_path, df

        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            with open(tmp_csv_path, 'wb') as f:
                f.write(uploaded_file.read())
            return tmp_csv_path, df

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        if os.path.exists(tmp_csv_path):
            os.remove(os.path.abspath(tmp_csv_path)) 
        return None, None
    
    return tmp_csv_path, df

@st.cache_data(show_spinner="Cacheando DataFrame na Memória...")
def get_execution_df_cached(temp_csv_path):
    """Lê o DataFrame e aplica amostragem, retornando o df, flag, e tamanho original."""
    df = pd.read_csv(temp_csv_path)
    original_rows = df.shape[0]
    is_sampled = False
    
    if original_rows > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=42)
        is_sampled = True
        
    return df, is_sampled, original_rows

# --- Inicialização do Agente (Corrigido para Caching) ---

@st.cache_resource
def get_agent_executor():
    """Cria e armazena o Agente Executor na cache."""
    llm = get_llm() # Chama o LLM cacheado
    if llm is None:
        return None

    # Prompt de Sistema (System Prompt)
    system_prompt = (
        "Você é um Multi Agente de IA SUPER ESPECIALISTA em Contabilidade e Análise de Dados. "
        "Sua função é analisar dados fornecidos e usar suas ferramentas para responder às perguntas. "
        "Use o histórico de chat para manter o contexto. Sempre use as ferramentas disponíveis para cálculos e plots. "
        "Responda de forma concisa e profissional. Não gere código Python a menos que seja para chamar uma ferramenta."
    )
    
    # Template do Chat (inclui System Prompt, Histórico, e Entrada do Usuário)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    # Criação do Agente (moderno, baseado em Tool Calling)
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Memória de Conversa (ajustada para LangChain moderna)
    memory = ConversationBufferWindowMemory(
        k=MAX_HISTORY_SIZE, 
        memory_key="chat_history", 
        return_messages=True, 
        input_key="input"
    )
    
    # Executor: orquestra o LLM, as ferramentas e a memória
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=False, 
        handle_parsing_errors=True,
        # O max_iterations é crucial para evitar loops infinitos de agentes
        max_iterations=15
    )
    
    return agent_executor

# --- Funções de Renderização e Parsing ---

def parse_and_display_response(response_obj):
    """
    Analisa a saída do Agente (AgentExecutor) e extrai o gráfico/texto.
    """
    output_text = response_obj["output"]
    PLOTLY_TAG = "<PLOTLY_JSON>"

    # 1. Tenta extrair e renderizar o gráfico
    if PLOTLY_TAG in output_text:
        try:
            # Extrai o JSON usando as tags
            json_str = output_text.split(PLOTLY_TAG, 1)[1].split("</PLOTLY_JSON>", 1)[0].strip()
            fig_dict = json.loads(json_str)
            fig = pio.from_json(json.dumps(fig_dict))
            
            # Renderiza o gráfico
            st.plotly_chart(fig, use_container_width=True)
            st.chat_message("assistant").markdown("✅ **Gráfico gerado:** Analise a visualização acima.")

            # Remove o JSON da saída de texto final
            output_text = output_text.split("PLOTLY_OUTPUT:", 1)[0].strip()

        except Exception as e:
            st.chat_message("assistant").error(f"⚠️ Falha ao renderizar o gráfico. Erro de JSON: {e}")
            
    # 2. Exibe o texto restante
    if output_text:
        st.chat_message("assistant").markdown(output_text)
        
    # 3. Adiciona a saída completa ao histórico de sessão (para relatórios)
    st.session_state.chat_history_list.append({"role": "assistant", "content": output_text})


# --- Layout e Interface ---

st.title("🤖 Multi Agente de Análise Fiscal e de Fraudes")
st.markdown("---")

# Inicialização de estado (mantendo compatibilidade)
if 'temp_csv_path' not in st.session_state: st.session_state.temp_csv_path = None
if 'df' not in st.session_state: st.session_state.df = None
if 'df_exec' not in st.session_state: st.session_state.df_exec = None
if 'chat_history_list' not in st.session_state: st.session_state.chat_history_list = []
if 'report_content' not in st.session_state: st.session_state.report_content = ""
if 'agent_executor' not in st.session_state: st.session_state.agent_executor = None

# Inicializa o executor do Agente
if 'agent_executor' not in st.session_state or st.session_state.agent_executor is None:
    # Chama a função sem parâmetros, resolvendo o erro de cache
    st.session_state.agent_executor = get_agent_executor()

# --- Barra Lateral (Upload e Relatório) ---
with st.sidebar:
    st.header("⚙️ Configurações de Análise")
    
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV, ZIP ou GZ:",
        type=['csv', 'zip', 'gz'],
        key="file_uploader"
    )
    
    st.subheader("Relatório Final")
    report_btn = st.button("📝 Gerar Conclusão da Análise", use_container_width=True)
    
    if st.session_state.report_content:
        st.download_button(
            label="⬇️ Baixar Relatório (Markdown)",
            data=st.session_state.report_content,
            file_name="relatorio_analise_ia.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- Processamento do Arquivo ---

if uploaded_file and st.session_state.temp_csv_path is None:
    # Passo 1: Lê o arquivo e salva o caminho temporário
    st.session_state.temp_csv_path, st.session_state.df = unzip_and_read_file(uploaded_file)
    
    if st.session_state.df is not None:
        # Passo 2: Lê o DataFrame cacheado (amostrado) para a execução do Agente
        # O resultado é salvo em df_exec, que é o que as tools usam
        st.session_state.df_exec, is_sampled, original_rows = get_execution_df_cached(st.session_state.temp_csv_path) 
        
        st.session_state.chat_history_list.clear()
        st.success(f"Arquivo '{uploaded_file.name}' carregado e pronto para análise!")
    else:
        st.error("Falha ao carregar o arquivo. Verifique o formato.")

# --- Processamento do Relatório ---

if report_btn and st.session_state.df is not None:
    # Acessa o executor do Agente
    agent_executor = st.session_state.agent_executor
    if agent_executor is None:
        st.error("Agente não inicializado. Verifique a chave da API.")
    else:
        # Pede ao Agente para fazer a conclusão
        final_prompt = "Faça uma conclusão resumida e completa de toda a análise de dados realizada no histórico, cobrindo as seções: Resumo Executivo, Detalhes da Análise, e Conclusão Final. Sua resposta deve ser SOMENTE o conteúdo do relatório em Markdown."
        
        with st.spinner("Gerando relatório completo..."):
            try:
                # O LangChain Tool Calling Agent usa o .invoke()
                response = agent_executor.invoke({"input": final_prompt})
                st.session_state.report_content = response['output']
                st.success("Relatório gerado com sucesso! Use o botão 'Baixar Relatório (Markdown)' na lateral.")
                
            except Exception as e:
                st.error(f"Erro ao gerar o relatório: {e}")

# --- Interface de Chat ---

# Exibe o histórico de chat
for item in st.session_state.chat_history_list:
    role = item['role']
    content = item['content']
    
    if role == "user":
        st.chat_message("user").markdown(content)
    else:
        # Renderiza a saída do agente (o LangChain Tool Agent já adiciona ao histórico)
        parse_and_display_response({"output": content}) 


# Campo de entrada de prompt do usuário
if st.session_state.df is not None and st.session_state.agent_executor:
    if prompt := st.chat_input("Faça sua pergunta ao Agente..."):
        
        agent_executor = st.session_state.agent_executor
        
        # Adiciona a pergunta ao histórico de exibição
        st.session_state.chat_history_list.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.spinner("Agente de IA está processando..."):
            try:
                # O LangChain Tool Calling Agent usa o .invoke()
                response = agent_executor.invoke({"input": prompt})

                # A saída do Agente deve ser tratada aqui
                output = response["output"]
                
                # A saída é processada e adicionada ao histórico dentro de parse_and_display_response
                parse_and_display_response({"output": output})
                
            except Exception as e:
                st.session_state.chat_history_list.append({"role": "assistant", "content": "Ocorreu um erro na comunicação com a IA. Por favor, tente novamente ou reformule sua pergunta."})
                st.chat_message("assistant").error(f"❌ Erro na execução: {e}. Tente novamente.")
                print(f"Erro na execução da API: {e}")

# Footer
if st.session_state.df is None:
    st.info("⚠️ Carregue um arquivo CSV, ZIP ou GZ para iniciar a análise.")
