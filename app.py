import streamlit as st
import pandas as pd
import os
import zipfile
import gzip
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from datetime import datetime
import time

# --- Configurações Iniciais ---
st.set_page_config(
    layout="wide", 
    page_title="Multi Agente de Análise Fiscal e de Fraudes",
    initial_sidebar_state="expanded"
)

# --- Constantes ---
MODEL_NAME = "gemini-2.0-flash-exp"  # Modelo mais econômico e rápido
MAX_TOKENS = 1024  # Limite de tokens por resposta
CACHE_TTL = 3600  # 1 hora de cache
MAX_DAILY_CALLS = 100  # Limite diário de chamadas API
MIN_CALL_INTERVAL = 2  # Segundos mínimos entre chamadas (rate limiting)

# Tenta obter a chave da API
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not API_KEY:
    st.error("❌ ERRO: Chave da API do Gemini não encontrada.")
    st.info("💡 Configure GEMINI_API_KEY nas secrets do Streamlit ou como variável de ambiente.")
    st.stop()

# --- Sistema de Cache Robusto ---
def get_cache_key(query: str, df_shape: tuple) -> str:
    """Gera chave única para cache"""
    cache_str = f"{query.lower().strip()}_{df_shape}"
    return hashlib.md5(cache_str.encode()).hexdigest()

def save_to_cache(key: str, value: str):
    """Salva resposta no cache com timestamp"""
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    st.session_state.cache[key] = {
        'value': value,
        'timestamp': time.time()
    }

def get_from_cache(key: str) -> str:
    """Recupera do cache se ainda válido"""
    if 'cache' not in st.session_state:
        return None
    
    cached = st.session_state.cache.get(key)
    if cached and (time.time() - cached['timestamp']) < CACHE_TTL:
        return cached['value']
    return None

# --- Sistema de Rate Limiting ---
def check_rate_limit() -> bool:
    """Verifica se pode fazer chamada (rate limiting)"""
    if 'last_call_time' not in st.session_state:
        st.session_state.last_call_time = 0
    
    time_since_last_call = time.time() - st.session_state.last_call_time
    
    if time_since_last_call < MIN_CALL_INTERVAL:
        wait_time = MIN_CALL_INTERVAL - time_since_last_call
        st.warning(f"⏱️ Aguarde {wait_time:.1f}s antes da próxima consulta...")
        time.sleep(wait_time)
    
    return True

def check_daily_limit() -> bool:
    """Verifica se atingiu limite diário"""
    # Reseta contador se mudou o dia
    today = datetime.now().strftime('%Y-%m-%d')
    
    if 'call_date' not in st.session_state or st.session_state.call_date != today:
        st.session_state.call_date = today
        st.session_state.api_calls = 0
    
    if st.session_state.api_calls >= MAX_DAILY_CALLS:
        return False
    
    return True

def record_api_call():
    """Registra chamada à API"""
    st.session_state.api_calls += 1
    st.session_state.last_call_time = time.time()

# --- Manipulação de Arquivos ---
@st.cache_data(show_spinner=False)
def load_file(uploaded_file) -> pd.DataFrame:
    """Carrega e processa arquivos CSV, ZIP ou GZ"""
    try:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        uploaded_file.seek(0)
        
        if file_extension == 'zip':
            with zipfile.ZipFile(uploaded_file, 'r') as zf:
                csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
                if not csv_files:
                    st.error("❌ Nenhum arquivo CSV encontrado no ZIP")
                    return None
                with zf.open(csv_files[0]) as csv_file:
                    return pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
        
        elif file_extension in ['gz', 'gzip']:
            with gzip.open(uploaded_file, 'rt', encoding='utf-8') as gz_file:
                return pd.read_csv(gz_file, low_memory=False)
        
        elif file_extension == 'csv':
            return pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
        
        st.error(f"❌ Formato não suportado: {file_extension}")
        return None
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        return None

# --- Sistema Inteligente de Gráficos ---
def should_create_chart(query: str) -> tuple:
    """Detecta se deve gerar gráfico e qual tipo"""
    query_lower = query.lower()
    
    # Palavras que indicam gráfico
    chart_triggers = {
        'bar': ['gráfico de barras', 'grafico de barras', 'gráfico de barra', 'chart bar', 
                'plote as', 'plote os', 'mostre em barras', 'visualize', 'visualização'],
        'line': ['gráfico de linha', 'grafico de linha', 'linha temporal', 'evolução', 
                 'tendência', 'ao longo do tempo', 'série temporal'],
        'scatter': ['dispersão', 'scatter', 'correlação', 'relação entre', 'comparação entre'],
        'pie': ['pizza', 'pie', 'proporção', 'percentual', 'distribuição percentual'],
        'histogram': ['histograma', 'distribuição', 'frequência'],
        'box': ['boxplot', 'box plot', 'outliers', 'quartis']
    }
    
    # Verifica se é uma solicitação de gráfico explícita
    for chart_type, triggers in chart_triggers.items():
        if any(trigger in query_lower for trigger in triggers):
            return True, chart_type
    
    # Palavras genéricas que indicam visualização
    if any(word in query_lower for word in ['gráfico', 'grafico', 'plot', 'plote', 'chart', 
                                              'visualiz', 'mostre graficamente', 'desenhe']):
        return True, 'auto'
    
    return False, None

def extract_columns_from_query(query: str, df: pd.DataFrame) -> list:
    """Extrai nomes de colunas mencionadas na query"""
    query_lower = query.lower()
    columns_found = []
    
    # Procura por colunas exatas
    for col in df.columns:
        if col.lower() in query_lower:
            columns_found.append(col)
    
    # Procura por padrões tipo "coluna X" ou "campo X"
    words = query_lower.split()
    for i, word in enumerate(words):
        if word in ['coluna', 'campo', 'variável', 'variavel'] and i + 1 < len(words):
            potential_col = words[i + 1].strip('.,;:')
            for col in df.columns:
                if potential_col in col.lower():
                    columns_found.append(col)
    
    return list(set(columns_found))  # Remove duplicatas

def create_smart_chart(df: pd.DataFrame, query: str, chart_type: str = 'auto'):
    """Cria gráficos inteligentemente baseado no contexto"""
    try:
        # Limita dados para performance
        df_sample = df.head(5000)
        
        # Identifica colunas
        mentioned_cols = extract_columns_from_query(query, df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Auto-detecta melhor tipo de gráfico
        if chart_type == 'auto':
            if len(mentioned_cols) >= 2:
                if all(col in numeric_cols for col in mentioned_cols[:2]):
                    chart_type = 'scatter'
                else:
                    chart_type = 'bar'
            elif len(mentioned_cols) == 1:
                if mentioned_cols[0] in numeric_cols:
                    chart_type = 'histogram'
                else:
                    chart_type = 'bar'
            elif categorical_cols:
                chart_type = 'bar'
            elif numeric_cols:
                chart_type = 'histogram'
            else:
                return None
        
        # GRÁFICO DE BARRAS
        if chart_type == 'bar':
            col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in categorical_cols else categorical_cols[0] if categorical_cols else None
            if not col:
                return None
            
            counts = df_sample[col].value_counts().head(15)
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={'x': col, 'y': 'Contagem'},
                title=f'📊 Top 15 - Distribuição de {col}',
                color=counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                height=500,
                hovermode='x'
            )
            return fig
        
        # GRÁFICO DE LINHA
        elif chart_type == 'line':
            if len(mentioned_cols) >= 2:
                x_col, y_col = mentioned_cols[0], mentioned_cols[1]
            elif len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
            else:
                return None
            
            data_sorted = df_sample[[x_col, y_col]].dropna().sort_values(x_col).head(500)
            fig = px.line(
                data_sorted,
                x=x_col,
                y=y_col,
                title=f'📈 {y_col} por {x_col}',
                markers=True
            )
            fig.update_layout(height=500, hovermode='x unified')
            return fig
        
        # GRÁFICO DE DISPERSÃO
        elif chart_type == 'scatter':
            if len(mentioned_cols) >= 2:
                x_col, y_col = mentioned_cols[0], mentioned_cols[1]
            elif len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
            else:
                return None
            
            data = df_sample[[x_col, y_col]].dropna().head(2000)
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                title=f'🔵 Correlação: {x_col} vs {y_col}',
                opacity=0.6,
                color=data[y_col],
                color_continuous_scale='plasma'
            )
            fig.update_layout(height=500)
            return fig
        
        # HISTOGRAMA
        elif chart_type == 'histogram':
            col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in numeric_cols else numeric_cols[0] if numeric_cols else None
            if not col:
                return None
            
            fig = px.histogram(
                df_sample,
                x=col,
                title=f'📊 Distribuição de {col}',
                nbins=50,
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(height=500, bargap=0.1)
            return fig
        
        # GRÁFICO DE PIZZA
        elif chart_type == 'pie':
            col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in categorical_cols else categorical_cols[0] if categorical_cols else None
            if not col:
                return None
            
            counts = df_sample[col].value_counts().head(10)
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                title=f'🥧 Proporção - {col}',
                hole=0.3
            )
            fig.update_layout(height=500)
            return fig
        
        # BOXPLOT
        elif chart_type == 'box':
            col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in numeric_cols else numeric_cols[0] if numeric_cols else None
            if not col:
                return None
            
            fig = px.box(
                df_sample,
                y=col,
                title=f'📦 Análise de Outliers - {col}',
                color_discrete_sequence=['#EF553B']
            )
            fig.update_layout(height=500)
            return fig
        
        return None
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao gerar gráfico: {str(e)}")
        return None

# --- Agente Otimizado ---
@st.cache_resource(show_spinner=False)
def create_agent(_df: pd.DataFrame):
    """Cria agente com configurações otimizadas"""
    
    # Prompt otimizado e direto
    system_prompt = """Você é um analista de dados Python especializado em análises fiscais.

REGRAS CRÍTICAS:
1. SEMPRE execute código Python para responder
2. Use 'df' como nome do DataFrame
3. Seja EXTREMAMENTE conciso - máximo 3-4 linhas de resposta
4. NUNCA explique o código, apenas mostre o resultado
5. Para perguntas simples, retorne APENAS o número/valor

Exemplos de respostas corretas:
- "O dataset tem 10.000 linhas"
- "A média é 1.234,56"
- "As top 5 categorias são: A (500), B (300), C (200), D (150), E (100)"

EXECUTE o código, não apenas mostre ele."""

    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=API_KEY,
            temperature=0,  # Zero temperatura = respostas determinísticas
            max_output_tokens=MAX_TOKENS,  # Limita tokens
            timeout=30,  # Timeout reduzido
            convert_system_message_to_human=True  # Compatibilidade Gemini
        )
        
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=_df,
            verbose=False,  # Reduz output desnecessário
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prefix=system_prompt,
            allow_dangerous_code=True,
            max_iterations=3,  # Reduzido de 4 para 3
            max_execution_time=25,
            early_stopping_method="generate",  # Para mais rápido
            handle_parsing_errors=True
        )
        
        return agent
        
    except Exception as e:
        st.error(f"❌ Erro ao criar agente: {str(e)}")
        return None

# --- Interface Principal ---
st.title("🤖 Multi Agente de Análise Fiscal e de Fraudes")
st.markdown("**Análise inteligente com economia de API e geração automática de gráficos**")
st.markdown("---")

# Inicialização de estados
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'report' not in st.session_state:
    st.session_state.report = ""
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'cache_hits' not in st.session_state:
    st.session_state.cache_hits = 0
if 'call_date' not in st.session_state:
    st.session_state.call_date = datetime.now().strftime('%Y-%m-%d')
if 'last_call_time' not in st.session_state:
    st.session_state.last_call_time = 0

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Métricas de uso
    col1, col2 = st.columns(2)
    with col1:
        calls_remaining = MAX_DAILY_CALLS - st.session_state.api_calls
        st.metric("🔥 Chamadas Hoje", st.session_state.api_calls, f"{calls_remaining} restantes")
    with col2:
        st.metric("💾 Cache Hits", st.session_state.cache_hits)
    
    # Economia e barra de progresso
    if st.session_state.api_calls > 0:
        total_queries = st.session_state.api_calls + st.session_state.cache_hits
        economia = (st.session_state.cache_hits / total_queries) * 100
        st.success(f"💰 Economia: {economia:.1f}%")
        
        # Barra de progresso do limite diário
        progress = st.session_state.api_calls / MAX_DAILY_CALLS
        st.progress(progress, text=f"Limite diário: {progress*100:.0f}%")
        
        # Alerta se próximo do limite
        if st.session_state.api_calls >= MAX_DAILY_CALLS * 0.8:
            st.warning(f"⚠️ Você usou {st.session_state.api_calls} de {MAX_DAILY_CALLS} chamadas")
    
    st.markdown("---")
    
    # Upload
    uploaded_file = st.file_uploader(
        "📁 Carregue seu arquivo",
        type=['csv', 'zip', 'gz'],
        help="Formatos: CSV, ZIP (com CSV) ou GZ"
    )
    
    st.markdown("---")
    
    # Relatório
    st.subheader("📊 Relatório")
    if st.button("🔄 Gerar Relatório", use_container_width=True):
        if st.session_state.agent:
            # Verifica limite diário
            if not check_daily_limit():
                st.error(f"❌ Limite diário de {MAX_DAILY_CALLS} chamadas atingido!")
                st.info("💡 O contador será resetado amanhã ou limpe o cache.")
            else:
                check_rate_limit()  # Rate limiting
                
                with st.spinner("Gerando relatório..."):
                    report_query = """Gere um relatório executivo com exatamente 5 tópicos:
1. Total de linhas e colunas
2. Principais colunas numéricas e suas médias
3. Top 3 categorias mais frequentes (se houver colunas categóricas)
4. Valores nulos por coluna (apenas se houver)
5. Uma anomalia ou insight relevante

Seja EXTREMAMENTE conciso. Use números."""
                    
                    try:
                        response = st.session_state.agent.invoke({"input": report_query})
                        st.session_state.report = response.get('output', str(response))
                        record_api_call()
                        st.success("✅ Relatório gerado!")
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)[:100]}")
    
    if st.session_state.report:
        st.download_button(
            "⬇️ Baixar Relatório",
            st.session_state.report,
            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Limpeza
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpar Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Total", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Configurações avançadas
    with st.expander("🔧 Configurações Avançadas"):
        st.info(f"**Limite Diário:** {MAX_DAILY_CALLS} chamadas")
        st.info(f"**Rate Limit:** {MIN_CALL_INTERVAL}s entre chamadas")
        st.info(f"**Cache TTL:** {CACHE_TTL/3600:.0f} hora(s)")
        st.info(f"**Max Tokens:** {MAX_TOKENS}")
        
        # Opção de exportar histórico
        if st.session_state.chat_history:
            chat_export = "\n\n".join([
                f"**{'Você' if role=='user' else 'Agente'}:** {msg}"
                for role, msg in st.session_state.chat_history
            ])
            st.download_button(
                "📥 Exportar Histórico",
                chat_export,
                file_name=f"historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# --- Processamento do Arquivo ---
if uploaded_file:
    if st.session_state.df is None:
        with st.spinner("📂 Carregando arquivo..."):
            df = load_file(uploaded_file)
            
            if df is not None:
                st.session_state.df = df
                
                with st.spinner("🤖 Inicializando agente..."):
                    st.session_state.agent = create_agent(df)
                
                if st.session_state.agent:
                    st.success(f"✅ **{uploaded_file.name}** carregado com sucesso!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Linhas", f"{df.shape[0]:,}")
                    with col2:
                        st.metric("📋 Colunas", df.shape[1])
                    with col3:
                        st.metric("💾 Memória", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                    
                    with st.expander("👀 Preview dos Dados", expanded=False):
                        st.dataframe(df.head(20), use_container_width=True)
                        
                        st.markdown("**📋 Tipos de Dados:**")
                        type_counts = df.dtypes.value_counts()
                        for dtype, count in type_counts.items():
                            st.write(f"- `{dtype}`: {count} colunas")

# --- Chat Interface ---
if st.session_state.agent and st.session_state.df is not None:
    
    # Exibe histórico
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
    
    # Input do usuário
    if user_query := st.chat_input("💬 Faça sua pergunta sobre os dados..."):
        
        # Verifica limite diário ANTES de processar
        if not check_daily_limit():
            st.error(f"❌ **Limite diário atingido!**")
            st.warning(f"Você já usou {MAX_DAILY_CALLS} chamadas hoje. O contador será resetado amanhã às 00:00.")
            st.info("💡 **Dica:** Use o cache! Refaça perguntas anteriores para economizar.")
            st.stop()
        
        # Adiciona ao histórico
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Verifica cache
        cache_key = get_cache_key(user_query, st.session_state.df.shape)
        cached_response = get_from_cache(cache_key)
        
        if cached_response:
            response_text = cached_response
            st.session_state.cache_hits += 1
            with st.chat_message("assistant"):
                st.markdown(response_text)
                st.caption("💾 Resposta do cache (0 tokens gastos)")
        else:
            # Aplica rate limiting
            check_rate_limit()
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analisando..."):
                    try:
                        # Consulta o agente
                        result = st.session_state.agent.invoke({
                            "input": user_query,
                            "handle_parsing_errors": True
                        })
                        
                        response_text = result.get('output', str(result)).strip()
                        
                        # Limpa resposta se muito verbosa
                        if len(response_text) > 500:
                            response_text = response_text[:500] + "..."
                        
                        st.markdown(response_text)
                        
                        # Salva no cache
                        save_to_cache(cache_key, response_text)
                        record_api_call()
                        
                        # Mostra tokens gastos aproximados
                        approx_tokens = len(user_query.split()) + len(response_text.split())
                        st.caption(f"🔥 ~{approx_tokens} tokens usados")
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "timeout" in error_msg.lower():
                            response_text = "⏱️ A consulta demorou muito. Tente uma pergunta mais simples."
                        elif "parsing" in error_msg.lower():
                            response_text = "⚠️ Erro ao interpretar resposta. Reformule sua pergunta de forma mais direta."
                        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                            response_text = "🚫 Limite da API Gemini atingido. Tente novamente mais tarde."
                        else:
                            response_text = f"❌ Erro: {error_msg[:200]}"
                        
                        st.error(response_text)
        
        # Adiciona resposta ao histórico
        st.session_state.chat_history.append(("assistant", response_text))
        
        # Detecta e gera gráfico
        should_chart, chart_type = should_create_chart(user_query)
        
        if should_chart:
            with st.spinner("📊 Gerando visualização..."):
                chart = create_smart_chart(st.session_state.df, user_query, chart_type)
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"chart_{len(st.session_state.chat_history)}")
                    st.success("✅ Gráfico gerado!")
                else:
                    st.info("💡 Especifique as colunas para gerar o gráfico (ex: 'gráfico de barras da coluna Status')")

else:
    # Mensagem inicial
    st.info("👆 **Carregue um arquivo CSV, ZIP ou GZ na barra lateral para começar**")
    
    st.markdown("### 💡 Exemplos de Análises Fiscais:")
    
    # Organiza exemplos por categoria
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 **Análise Descritiva**")
        st.markdown("""
        - Quantas notas fiscais foram emitidas no período?
        - Qual o valor total das transações?
        - Mostre a média e mediana do valor das notas
        - Qual a distribuição dos valores das notas fiscais?
        - Existem valores atípicos (outliers) nos valores?
        """)
        
        st.markdown("#### 🔍 **Detecção de Anomalias**")
        st.markdown("""
        - Identifique notas fiscais com valores suspeitos
        - Existem transações duplicadas?
        - Há notas emitidas fora do horário comercial?
        - Quais fornecedores têm maior variabilidade de valores?
        """)
    
    with col2:
        st.markdown("#### 📈 **Análise Temporal**")
        st.markdown("""
        - Gráfico de linha da evolução mensal das vendas
        - Qual o dia da semana com mais transações?
        - Existe sazonalidade nos dados fiscais?
        - Mostre a tendência de crescimento mês a mês
        """)
        
        st.markdown("#### 🎯 **Análise de Correlação**")
        st.markdown("""
        - Gráfico de dispersão entre quantidade e valor
        - Qual a correlação entre valor unitário e quantidade?
        - Existe relação entre fornecedor e valor médio?
        - Mostre a matriz de correlação das variáveis numéricas
        """)
    
    st.markdown("---")
    st.markdown("### 🎓 **Guia de Análise Fiscal Completa**")
    
    with st.expander("📋 Roteiro de Análise Exploratória", expanded=False):
        st.markdown("""
        **1️⃣ ENTENDIMENTO DOS DADOS**
        - Quais são as colunas do dataset?
        - Quantas linhas e colunas existem?
        - Quais são os tipos de dados (numéricos, categóricos)?
        - Mostre as primeiras 10 linhas
        
        **2️⃣ QUALIDADE DOS DADOS**
        - Existem valores nulos? Em quais colunas?
        - Existem dados duplicados?
        - Qual o percentual de completude dos dados?
        
        **3️⃣ ANÁLISE ESTATÍSTICA**
        - Quais as medidas de tendência central (média, mediana, moda)?
        - Qual a variabilidade dos dados (desvio padrão, variância)?
        - Gráfico de distribuição das variáveis numéricas
        - Identifique valores atípicos (boxplot)
        
        **4️⃣ ANÁLISE TEMPORAL**
        - Existe tendência temporal nos dados?
        - Gráfico de linha da evolução ao longo do tempo
        - Qual o período com maior/menor atividade?
        
        **5️⃣ ANÁLISE DE CORRELAÇÃO**
        - Existe correlação entre as variáveis numéricas?
        - Gráfico de dispersão entre variáveis chave
        - Matriz de correlação (heatmap)
        
        **6️⃣ ANÁLISE CATEGÓRICA**
        - Gráfico de barras das top 10 categorias
        - Qual a distribuição percentual por categoria?
        - Tabela cruzada entre categorias importantes
        
        **7️⃣ DETECÇÃO DE FRAUDES**
        - Identificar padrões suspeitos ou anomalias
        - Transações fora do padrão estatístico
        - Análise de fornecedores com comportamento atípico
        """)
    
    st.markdown("---")
    st.info("💡 **Dica:** Após carregar seus dados, use o botão '🎯 Análise Rápida' na sidebar para um diagnóstico automático!")

# --- Footer ---
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🤖 Powered by Gemini 2.0 Flash")
with col2:
    if st.session_state.api_calls > 0:
        st.caption(f"📊 {st.session_state.api_calls} chamadas hoje")
with col3:
    if st.session_state.cache_hits > 0:
        st.caption(f"💾 {st.session_state.cache_hits} cache hits")