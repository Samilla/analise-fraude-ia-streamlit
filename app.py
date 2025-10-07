import streamlit as st
import pandas as pd
import os
import tempfile
import zipfile
import gzip
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import hashlib

# --- Configurações Iniciais do Streamlit ---
st.set_page_config(layout="wide", page_title="Multi Agente de Análise Fiscal e de Fraudes")

# --- Constantes e Variáveis Globais ---
pio.templates.default = "plotly_white"

# CORREÇÃO 1: Modelo correto do Gemini
MODEL_NAME = "gemini-1.5-flash"  # Modelo correto e mais econômico

# Tenta obter a chave da API
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_KEY", "")

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada.")

# CORREÇÃO 2: Cache para respostas (economia de API)
@st.cache_data(ttl=3600)
def cache_query_response(query_hash, _df):
    """Cache de respostas para economizar chamadas à API"""
    return None

# --- Funções de Manipulação de Arquivos ---
def unzip_and_read_file(uploaded_file):
    """Descompacta e lê arquivos CSV, ZIP ou GZ"""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    uploaded_file.seek(0)
    
    try:
        if file_extension == 'zip':
            with zipfile.ZipFile(uploaded_file, 'r') as zf:
                csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
                if not csv_files:
                    st.error("Nenhum arquivo CSV encontrado dentro do ZIP.")
                    return None
                with zf.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)
                    return df
                    
        elif uploaded_file.name.endswith(('.gz', '.gzip')):
            with gzip.open(uploaded_file, 'rt') as gz_file:
                df = pd.read_csv(gz_file)
                return df
                
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            return df
            
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None
    
    return None

# CORREÇÃO 3: Função para criar gráficos automaticamente
def create_chart_from_query(df, query, response_text):
    """
    Cria gráficos automaticamente baseado em palavras-chave na query
    """
    query_lower = query.lower()
    
    # Detecta tipo de gráfico solicitado
    if any(word in query_lower for word in ['gráfico', 'grafico', 'visualiz', 'plot', 'chart']):
        try:
            # Gráfico de barras
            if any(word in query_lower for word in ['barra', 'bar', 'contagem', 'frequência']):
                # Pega primeira coluna categórica
                cat_col = df.select_dtypes(include=['object']).columns[0]
                counts = df[cat_col].value_counts().head(10)
                fig = px.bar(x=counts.index, y=counts.values, 
                           labels={'x': cat_col, 'y': 'Contagem'},
                           title=f'Top 10 - {cat_col}')
                return fig
            
            # Gráfico de linha (temporal)
            elif any(word in query_lower for word in ['linha', 'line', 'tempo', 'time', 'tendência']):
                # Procura coluna de data ou numérica sequencial
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    fig = px.line(df.head(100), x=numeric_cols[0], y=numeric_cols[1],
                                title=f'{numeric_cols[1]} ao longo de {numeric_cols[0]}')
                    return fig
            
            # Gráfico de dispersão (correlação)
            elif any(word in query_lower for word in ['dispersão', 'scatter', 'correlação', 'relação']):
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df.head(1000), x=numeric_cols[0], y=numeric_cols[1],
                                   title=f'Relação entre {numeric_cols[0]} e {numeric_cols[1]}')
                    return fig
            
            # Histograma
            elif any(word in query_lower for word in ['histograma', 'histogram', 'distribuição']):
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 1:
                    fig = px.histogram(df, x=numeric_cols[0],
                                     title=f'Distribuição de {numeric_cols[0]}')
                    return fig
                    
        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico automaticamente: {e}")
            return None
    
    return None

# --- Funções do Agente ---
@st.cache_resource
def load_llm_and_agent(_df):
    """
    CORREÇÃO 4: Usa Pandas Agent ao invés de CSV Agent
    Permite execução de código Python para análises complexas
    """
    
    # Prompt otimizado e mais direto
    analyst_prompt = """Você é um Analista de Dados Fiscal especializado.

REGRAS IMPORTANTES:
1. Seja CONCISO e DIRETO - respostas com no máximo 3 parágrafos
2. Use APENAS os dados do DataFrame fornecido
3. Para cálculos, use pandas diretamente (df.groupby, df.corr, etc)
4. NÃO tente criar gráficos - apenas descreva os dados
5. Formate números com 2 casas decimais

ANÁLISES PERMITIDAS:
- Estatísticas descritivas
- Agrupamentos e agregações
- Correlações
- Identificação de outliers
- Análise de valores faltantes"""

    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=API_KEY,
            temperature=0.1,  # Baixa temperatura = respostas mais consistentes
            max_output_tokens=800,  # CORREÇÃO 5: Limita tokens para economizar
            timeout=60
        )
        
        # CORREÇÃO 6: Usa create_pandas_dataframe_agent
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=_df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prefix=analyst_prompt,
            allow_dangerous_code=True,
            max_iterations=3,  # CORREÇÃO 7: Limita iterações
            max_execution_time=30,  # Timeout por query
            handle_parsing_errors=True
        )
        
        return agent
        
    except Exception as e:
        st.error(f"Erro ao criar agente: {e}")
        return None

# CORREÇÃO 8: Sistema de cache de queries
def get_query_hash(query, df_shape):
    """Gera hash único para a query"""
    query_str = f"{query}_{df_shape}"
    return hashlib.md5(query_str.encode()).hexdigest()

# --- Layout e Interface ---
st.title("🤖 Multi Agente de Análise Fiscal e de Fraudes")
st.markdown("---")

# Inicialização de estados
if 'data_agent' not in st.session_state:
    st.session_state.data_agent = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history_list' not in st.session_state:
    st.session_state.chat_history_list = []
if 'report_content' not in st.session_state:
    st.session_state.report_content = ""
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}
if 'api_calls_count' not in st.session_state:
    st.session_state.api_calls_count = 0

# --- Barra Lateral ---
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Contador de chamadas API
    st.metric("Chamadas à API", st.session_state.api_calls_count)
    
    # Upload
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo:",
        type=['csv', 'zip', 'gz'],
        key="file_uploader"
    )
    
    # Botão de relatório
    st.subheader("📊 Relatório")
    report_btn = st.button("Gerar Relatório Completo", use_container_width=True)
    
    # Download
    if st.session_state.report_content:
        st.download_button(
            label="⬇️ Baixar Relatório",
            data=st.session_state.report_content,
            file_name="relatorio_analise.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    # Botão para limpar cache
    if st.button("🔄 Limpar Cache", use_container_width=True):
        st.session_state.query_cache = {}
        st.session_state.api_calls_count = 0
        st.cache_data.clear()
        st.success("Cache limpo!")

# --- Processamento do Arquivo ---
if uploaded_file and st.session_state.df is None:
    df = unzip_and_read_file(uploaded_file)
    
    if df is not None:
        st.session_state.df = df
        
        with st.spinner("Inicializando agente..."):
            st.session_state.data_agent = load_llm_and_agent(df)
        
        if st.session_state.data_agent:
            st.success(f"✅ Arquivo '{uploaded_file.name}' carregado!")
            st.info(f"📊 Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
            
            # Preview dos dados
            with st.expander("👀 Preview dos Dados"):
                st.dataframe(df.head(10), use_container_width=True)
                st.write("**Colunas:**", ", ".join(df.columns.tolist()))

# --- Processamento do Relatório ---
if report_btn and st.session_state.data_agent and st.session_state.df is not None:
    report_prompt = """Gere um relatório executivo conciso com:

1. **Resumo dos Dados** (3 linhas)
2. **Principais Estatísticas** (métricas chave)
3. **Insights Importantes** (2-3 descobertas)
4. **Recomendações** (2-3 ações)

Use Markdown. Seja BREVE."""

    with st.spinner("Gerando relatório..."):
        try:
            response = st.session_state.data_agent.invoke({"input": report_prompt})
            st.session_state.report_content = response['output']
            st.session_state.api_calls_count += 1
            st.success("✅ Relatório gerado!")
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {e}")

# --- Interface de Chat ---
for role, message in st.session_state.chat_history_list:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)

# Campo de entrada
if st.session_state.data_agent and st.session_state.df is not None:
    if prompt := st.chat_input("Faça sua pergunta..."):
        # Adiciona pergunta ao histórico
        st.session_state.chat_history_list.append(("user", prompt))
        st.chat_message("user").markdown(prompt)
        
        # Verifica cache
        query_hash = get_query_hash(prompt, st.session_state.df.shape)
        
        if query_hash in st.session_state.query_cache:
            response_text = st.session_state.query_cache[query_hash]
            st.info("💾 Resposta do cache (economia de API)")
        else:
            with st.spinner("🤖 Processando..."):
                try:
                    # Executa query
                    response = st.session_state.data_agent.invoke({"input": prompt})
                    response_text = response['output']
                    
                    # Salva no cache
                    st.session_state.query_cache[query_hash] = response_text
                    st.session_state.api_calls_count += 1
                    
                except Exception as e:
                    response_text = f"⚠️ Erro ao processar: {str(e)[:200]}"
                    st.error(response_text)
        
        # Adiciona resposta ao histórico
        st.session_state.chat_history_list.append(("agent", response_text))
        st.chat_message("assistant").markdown(response_text)
        
        # CORREÇÃO 9: Gera gráfico automaticamente se solicitado
        chart = create_chart_from_query(st.session_state.df, prompt, response_text)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

else:
    st.info("⚠️ Carregue um arquivo CSV, ZIP ou GZ para iniciar.")