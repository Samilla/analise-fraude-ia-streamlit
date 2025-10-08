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

# CORREÇÃO 1: Modelo Gemini 2.5 Flash
MODEL_NAME = "gemini-2.5-flash"  # Modelo mais recente

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

# CORREÇÃO 3: Sistema inteligente de detecção e geração de gráficos
def detect_chart_request(query):
    """Detecta se o usuário quer um gráfico e qual tipo"""
    query_lower = query.lower()
    
    # IMPORTANTE: Palavras que NÃO indicam gráfico (queries de informação)
    info_keywords = ['coluna', 'colunas', 'linhas', 'registros', 'quantas', 'quantos', 
                     'primeiras', 'últimas', 'tipo', 'tipos', 'estrutura', 'formato',
                     'nome', 'nomes', 'lista', 'listar', 'informações', 'informacao']
    
    # Se for uma query de informação, NÃO gera gráfico
    if any(word in query_lower for word in info_keywords):
        # Exceção: se explicitamente pedir gráfico junto
        if not any(word in query_lower for word in ['gráfico', 'grafico', 'plot', 'chart', 'visualiz']):
            return None, None
    
    # Palavras-chave que indicam solicitação de gráfico (precisa ser EXPLÍCITO)
    chart_keywords = ['gráfico', 'grafico', 'visualiz', 'plot', 'plote', 'chart', 
                      'desenhe', 'crie um gráfico', 'faça um gráfico', 'gere um gráfico']
    
    wants_chart = any(word in query_lower for word in chart_keywords)
    
    if not wants_chart:
        return None, None
    
    # Detecta tipo de gráfico
    if any(word in query_lower for word in ['barra', 'bar', 'barras', 'contagem', 'frequência', 'top']):
        return 'bar', None
    elif any(word in query_lower for word in ['linha', 'line', 'temporal', 'tempo', 'time', 'evolução', 'tendência']):
        return 'line', None
    elif any(word in query_lower for word in ['dispersão', 'scatter', 'correlação', 'correlacao', 'relação', 'relacao']):
        return 'scatter', None
    elif any(word in query_lower for word in ['histograma', 'histogram', 'distribuição', 'distribuicao']):
        return 'histogram', None
    elif any(word in query_lower for word in ['pizza', 'pie', 'proporção', 'proporcao', 'percentual']):
        return 'pie', None
    elif any(word in query_lower for word in ['boxplot', 'box', 'caixa', 'outlier']):
        return 'box', None
    else:
        return 'auto', None  # Escolhe automaticamente

def extract_columns_from_query(query, df):
    """Extrai nomes de colunas mencionadas na query"""
    columns_found = []
    query_lower = query.lower()
    
    for col in df.columns:
        if col.lower() in query_lower:
            columns_found.append(col)
    
    return columns_found

def create_chart_from_query(df, query, chart_type=None):
    """
    Cria gráficos baseado no tipo detectado e nas colunas do DataFrame
    """
    try:
        # Extrai colunas mencionadas na query
        mentioned_cols = extract_columns_from_query(query, df)
        
        # Separa colunas por tipo
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Se mencionou colunas específicas, usa elas
        if mentioned_cols:
            x_col = mentioned_cols[0] if len(mentioned_cols) > 0 else None
            y_col = mentioned_cols[1] if len(mentioned_cols) > 1 else None
        else:
            x_col = None
            y_col = None
        
        # GRÁFICO DE BARRAS
        if chart_type == 'bar':
            if x_col and x_col in categorical_cols:
                counts = df[x_col].value_counts().head(15)
            elif categorical_cols:
                x_col = categorical_cols[0]
                counts = df[x_col].value_counts().head(15)
            else:
                return None
            
            fig = px.bar(
                x=counts.index, 
                y=counts.values,
                labels={'x': x_col, 'y': 'Contagem'},
                title=f'📊 Distribuição de {x_col}',
                color=counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            return fig
        
        # GRÁFICO DE LINHA
        elif chart_type == 'line':
            if x_col and y_col and x_col in numeric_cols and y_col in numeric_cols:
                data = df[[x_col, y_col]].head(500)
            elif len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                data = df[[x_col, y_col]].head(500)
            else:
                return None
            
            fig = px.line(
                data, 
                x=x_col, 
                y=y_col,
                title=f'📈 {y_col} ao longo de {x_col}',
                markers=True
            )
            return fig
        
        # GRÁFICO DE DISPERSÃO
        elif chart_type == 'scatter':
            if x_col and y_col and x_col in numeric_cols and y_col in numeric_cols:
                data = df[[x_col, y_col]].head(1000)
            elif len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                data = df[[x_col, y_col]].head(1000)
            else:
                return None
            
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                title=f'🔵 Relação entre {x_col} e {y_col}',
                opacity=0.6,
                color=data[y_col] if y_col else None
            )
            return fig
        
        # HISTOGRAMA
        elif chart_type == 'histogram':
            if x_col and x_col in numeric_cols:
                col = x_col
            elif numeric_cols:
                col = numeric_cols[0]
            else:
                return None
            
            fig = px.histogram(
                df,
                x=col,
                title=f'📊 Distribuição de {col}',
                nbins=30,
                color_discrete_sequence=['#636EFA']
            )
            return fig
        
        # GRÁFICO DE PIZZA
        elif chart_type == 'pie':
            if x_col and x_col in categorical_cols:
                col = x_col
            elif categorical_cols:
                col = categorical_cols[0]
            else:
                return None
            
            counts = df[col].value_counts().head(10)
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                title=f'🥧 Proporção de {col}'
            )
            return fig
        
        # BOXPLOT
        elif chart_type == 'box':
            if y_col and y_col in numeric_cols:
                col = y_col
            elif numeric_cols:
                col = numeric_cols[0]
            else:
                return None
            
            fig = px.box(
                df,
                y=col,
                title=f'📦 Boxplot de {col}'
            )
            return fig
        
        # AUTO: Escolhe o melhor gráfico baseado nos dados
        elif chart_type == 'auto':
            if len(categorical_cols) > 0:
                return create_chart_from_query(df, query, 'bar')
            elif len(numeric_cols) >= 2:
                return create_chart_from_query(df, query, 'scatter')
            elif len(numeric_cols) == 1:
                return create_chart_from_query(df, query, 'histogram')
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao gerar gráfico: {e}")
        return None
    
    return None

# --- Funções do Agente ---
@st.cache_resource
def load_llm_and_agent(_df):
    """
    CORREÇÃO 4: Usa Pandas Agent ao invés de CSV Agent
    Permite execução de código Python para análises complexas
    """
    
    # CORREÇÃO: Prompt simplificado que força execução de código
    analyst_prompt = """Você é um assistente Python especializado em análise de dados.

REGRAS OBRIGATÓRIAS:
1. SEMPRE execute código Python para responder
2. NUNCA retorne apenas o código, EXECUTE-O
3. Use 'df' como nome do DataFrame
4. Seja direto e objetivo

Quando o usuário perguntar algo, você DEVE:
- Executar código Python
- Retornar o resultado da execução
- NÃO apenas mostrar o código

IMPORTANTE: Use python_repl_ast para executar o código."""

    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=API_KEY,
            temperature=0.1,  # Baixa temperatura = respostas mais consistentes
            max_output_tokens=800,  # CORREÇÃO 5: Limita tokens para economizar
            timeout=60
        )
        
        # CORREÇÃO 6: Configuração robusta do agente com tratamento de erros
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=_df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prefix=analyst_prompt,
            allow_dangerous_code=True,
            max_iterations=4,
            max_execution_time=45,
            handle_parsing_errors=True  # Ativa tratamento de erros
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
                    # CORREÇÃO: Tratamento robusto de erros
                    response = st.session_state.data_agent.invoke({
                        "input": prompt,
                        "handle_parsing_errors": True
                    })
                    
                    # Extrai o output (pode vir em diferentes formatos)
                    if isinstance(response, dict):
                        response_text = response.get('output', str(response))
                    else:
                        response_text = str(response)
                    
                    # Limpa a resposta
                    response_text = response_text.strip()
                    
                    # Salva no cache
                    st.session_state.query_cache[query_hash] = response_text
                    st.session_state.api_calls_count += 1
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Trata erros de parsing especificamente
                    if "parsing error" in error_msg.lower():
                        response_text = """⚠️ **Erro de interpretação da resposta.**
                        
Tente reformular sua pergunta de forma mais direta, como:
- "Quantas linhas tem o dataset?"
- "Mostre as 5 primeiras linhas"
- "Qual a média da coluna X?"
- "Quais são as colunas?"
"""
                    else:
                        response_text = f"⚠️ Erro ao processar: {error_msg[:200]}"
                    
                    st.error(response_text)
        
        # Adiciona resposta ao histórico
        st.session_state.chat_history_list.append(("agent", response_text))
        st.chat_message("assistant").markdown(response_text)
        
        # CORREÇÃO 9: Detecta e gera gráfico se solicitado
        chart_type, _ = detect_chart_request(prompt)
        
        if chart_type:
            st.info("📊 Gerando visualização...")
            chart = create_chart_from_query(st.session_state.df, prompt, chart_type)
            
            if chart:
                st.plotly_chart(chart, use_container_width=True)
                st.success("✅ Gráfico gerado com sucesso!")
            else:
                st.warning("⚠️ Não foi possível gerar o gráfico. Tente especificar as colunas na sua pergunta.")

else:
    st.info("⚠️ Carregue um arquivo CSV, ZIP ou GZ para iniciar.")