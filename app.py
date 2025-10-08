import streamlit as st
import pandas as pd
import os
import zipfile
import gzip
import json
from langchain_google_genai import ChatGoogleGenerativeAI
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import hashlib
import numpy as np

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Multi Agente de Análise Fiscal e de Fraudes")
pio.templates.default = "plotly_white"

# Modelo - Testando versões disponíveis
MODEL_NAME = "gemini-2.5-flash"  # Modelo mais estável e universalmente disponível

# API Key
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_KEY", "")

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada.")

# --- Funções de Análise Direta (SEM AGENT) ---
def analyze_dataframe(df):
    """Retorna análise básica do DataFrame"""
    analysis = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
        'sample': df.head(5).to_dict('records')
    }
    return analysis

def execute_query_directly(df, query):
    """
    Executa queries diretamente no DataFrame sem usar agent
    Muito mais estável e rápido
    """
    query_lower = query.lower()
    
    try:
        # Lista colunas
        if any(word in query_lower for word in ['coluna', 'colunas', 'lista', 'liste']):
            result = f"**Colunas do Dataset:**\n\n" + "\n".join([f"- {col}" for col in df.columns])
            return result, None
        
        # Conta linhas
        elif any(word in query_lower for word in ['quantas linhas', 'total de linhas', 'linhas tem', 'count', 'tamanho']):
            result = f"📊 **Total de linhas:** {len(df):,}\n**Total de colunas:** {len(df.columns)}"
            return result, None
        
        # Primeiras linhas
        elif any(word in query_lower for word in ['primeira', 'primeiras', 'head', 'mostr']):
            n = 5
            for word in query_lower.split():
                if word.isdigit():
                    n = int(word)
            result = f"**Primeiras {n} linhas:**\n\n"
            result += df.head(n).to_markdown(index=False)
            return result, None
        
        # Estatísticas
        elif any(word in query_lower for word in ['estatística', 'estatistica', 'describe', 'resumo']):
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) > 0:
                result = "📊 **Estatísticas Descritivas:**\n\n"
                result += numeric_df.describe().to_markdown()
                return result, None
            else:
                return "⚠️ Não há colunas numéricas no dataset.", None
        
        # Valores únicos
        elif 'único' in query_lower or 'unicos' in query_lower or 'unique' in query_lower:
            for col in df.columns:
                if col.lower() in query_lower:
                    unique_vals = df[col].nunique()
                    result = f"📊 **Coluna '{col}':**\n\n"
                    result += f"- Valores únicos: {unique_vals:,}\n"
                    if unique_vals <= 20:
                        result += f"\n**Valores:**\n" + "\n".join([f"- {val}" for val in df[col].unique()])
                    return result, None
        
        # Correlação
        elif 'correlação' in query_lower or 'correlacao' in query_lower:
            cols_mentioned = [col for col in df.columns if col.lower() in query_lower]
            
            if len(cols_mentioned) >= 2:
                col1, col2 = cols_mentioned[0], cols_mentioned[1]
                if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                    corr = df[col1].corr(df[col2])
                    
                    # Interpretação
                    if abs(corr) > 0.7:
                        strength = "forte"
                    elif abs(corr) > 0.4:
                        strength = "moderada"
                    else:
                        strength = "fraca"
                    
                    direction = "positiva" if corr > 0 else "negativa"
                    
                    result = f"📊 **Correlação entre {col1} e {col2}:**\n\n"
                    result += f"- Valor: **{corr:.4f}**\n\n"
                    result += f"**📈 Interpretação:**\n"
                    result += f"A correlação é **{strength} e {direction}**. "
                    
                    if abs(corr) > 0.7:
                        result += f"Isso indica uma relação linear forte entre as variáveis."
                    elif abs(corr) > 0.4:
                        result += f"Existe uma relação moderada entre as variáveis."
                    else:
                        result += f"Há pouca ou nenhuma relação linear entre as variáveis."
                    
                    return result, ('scatter', col1, col2)
            else:
                # Matriz de correlação geral
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) >= 2:
                    corr_matrix = numeric_df.corr()
                    result = "📊 **Matriz de Correlação:**\n\n"
                    result += corr_matrix.to_markdown()
                    return result, ('heatmap', None, None)
        
        # Média
        elif 'média' in query_lower or 'media' in query_lower or 'mean' in query_lower:
            for col in df.columns:
                if col.lower() in query_lower:
                    if df[col].dtype in ['int64', 'float64']:
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        result = f"📊 **Estatísticas de '{col}':**\n\n"
                        result += f"- Média: **{mean_val:.2f}**\n"
                        result += f"- Desvio Padrão: **{std_val:.2f}**\n"
                        result += f"- Mínimo: {df[col].min():.2f}\n"
                        result += f"- Máximo: {df[col].max():.2f}\n"
                        return result, ('histogram', col, None)
        
        # Valores faltantes
        elif 'faltante' in query_lower or 'nulo' in query_lower or 'null' in query_lower or 'missing' in query_lower:
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                result = "⚠️ **Valores Faltantes:**\n\n"
                for col, count in missing.items():
                    pct = (count / len(df)) * 100
                    result += f"- **{col}**: {count:,} ({pct:.2f}%)\n"
            else:
                result = "✅ **Não há valores faltantes no dataset!**"
            return result, None
        
        # Distribuição
        elif 'distribuição' in query_lower or 'distribuicao' in query_lower:
            for col in df.columns:
                if col.lower() in query_lower:
                    if df[col].dtype in ['int64', 'float64']:
                        return f"📊 Distribuição de **{col}**", ('histogram', col, None)
                    else:
                        counts = df[col].value_counts()
                        result = f"📊 **Distribuição de '{col}':**\n\n"
                        result += counts.head(10).to_markdown()
                        return result, ('bar', col, None)
        
        # Fallback: usa LLM apenas para interpretação
        else:
            return None, None
            
    except Exception as e:
        return f"❌ Erro ao processar: {str(e)}", None

# --- Funções de Gráficos ---
def create_chart(df, chart_info):
    """Cria gráficos baseado no tipo e colunas"""
    if not chart_info:
        return None
    
    chart_type, col1, col2 = chart_info
    
    try:
        if chart_type == 'scatter' and col1 and col2:
            fig = px.scatter(df.head(1000), x=col1, y=col2,
                           title=f'📊 Relação entre {col1} e {col2}',
                           opacity=0.6)
            fig.update_layout(showlegend=False)
            return fig
        
        elif chart_type == 'histogram' and col1:
            fig = px.histogram(df, x=col1, nbins=30,
                             title=f'📊 Distribuição de {col1}')
            fig.update_layout(showlegend=False)
            return fig
        
        elif chart_type == 'bar' and col1:
            counts = df[col1].value_counts().head(15)
            fig = px.bar(x=counts.index, y=counts.values,
                       labels={'x': col1, 'y': 'Contagem'},
                       title=f'📊 Top 15 - {col1}')
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            return fig
        
        elif chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                          title='📊 Matriz de Correlação',
                          color_continuous_scale='RdBu_r',
                          labels=dict(color="Correlação"))
            return fig
            
    except Exception as e:
        st.warning(f"⚠️ Erro ao criar gráfico: {e}")
        return None
    
    return None

def use_llm_for_complex_query(df, query):
    """Usa LLM apenas para queries complexas que não podem ser resolvidas diretamente"""
    
    # Lista de modelos para tentar (em ordem de prioridade)
    models_to_try = [
        "gemini-pro",
        "models/gemini-pro",
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro"
    ]
    
    # Prepara contexto do DataFrame
    df_info = f"""Dataset Info:
- Linhas: {len(df)}
- Colunas: {', '.join(df.columns.tolist())}
- Tipos: {df.dtypes.to_dict()}

Primeiras 3 linhas:
{df.head(3).to_string()}

Estatísticas básicas:
{df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else 'Sem colunas numéricas'}
"""
    
    prompt = f"""{df_info}

Pergunta do usuário: {query}

Instruções:
1. Analise os dados fornecidos
2. Responda de forma clara e objetiva
3. Use markdown para formatação
4. Se aplicável, sugira visualizações

Resposta:"""
    
    # Tenta cada modelo até um funcionar
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=API_KEY,
                temperature=0.3,
                max_output_tokens=800,
                timeout=60
            )
            
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            error_msg = str(e)
            # Se for erro 404, tenta próximo modelo
            if "404" in error_msg or "not found" in error_msg.lower():
                continue
            # Se for outro erro, retorna mensagem
            else:
                return f"❌ Erro ao usar LLM: {error_msg[:200]}"
    
    # Se nenhum modelo funcionou
    return """⚠️ **Não foi possível acessar o modelo Gemini.**

**Soluções:**
1. Verifique sua API Key em: https://makersuite.google.com/app/apikey
2. Teste qual modelo está disponível com o script de teste
3. Use perguntas diretas que não precisam de LLM:
   - "Liste as colunas"
   - "Mostre estatísticas"
   - "Correlação entre X e Y"

💡 A maioria das análises funcionam SEM precisar do LLM!"""

# --- Funções de Arquivo ---
def unzip_and_read_file(uploaded_file):
    """Descompacta e lê arquivos"""
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

# --- Interface ---
st.title("🤖 Multi Agente de Análise Fiscal e de Fraudes")
st.markdown("---")

# Estados
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    st.metric("Perguntas realizadas", st.session_state.query_count)
    
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo:",
        type=['csv', 'zip', 'gz'],
        key="file_uploader"
    )
    
    if st.button("🔄 Reiniciar", use_container_width=True):
        st.session_state.df = None
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.rerun()

# Processamento do arquivo
if uploaded_file and st.session_state.df is None:
    with st.spinner("Carregando arquivo..."):
        df = unzip_and_read_file(uploaded_file)
        
        if df is not None:
            st.session_state.df = df
            st.success(f"✅ Arquivo '{uploaded_file.name}' carregado!")
            st.info(f"📊 Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
            st.info('💡 **Para iniciar, pergunte: "Liste as colunas"**')
            
            with st.expander("👀 Preview dos Dados"):
                st.dataframe(df.head(10), width='stretch')

# Chat
for role, message in st.session_state.chat_history:
    st.chat_message(role).markdown(message)

# Input
if st.session_state.df is not None:
    if prompt := st.chat_input("Faça sua pergunta..."):
        # Adiciona pergunta
        st.session_state.chat_history.append(("user", prompt))
        st.chat_message("user").markdown(prompt)
        
        with st.spinner("🤖 Analisando..."):
            # Tenta análise direta (rápida e sem API)
            result, chart_info = execute_query_directly(st.session_state.df, prompt)
            
            # Se não conseguiu responder diretamente, usa LLM
            if result is None:
                result = use_llm_for_complex_query(st.session_state.df, prompt)
                st.session_state.query_count += 1
            
            # Exibe resposta
            st.session_state.chat_history.append(("assistant", result))
            st.chat_message("assistant").markdown(result)
            
            # Cria gráfico se aplicável
            if chart_info:
                chart = create_chart(st.session_state.df, chart_info)
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"chart_{len(st.session_state.chat_history)}")

else:
    st.info("⚠️ Carregue um arquivo CSV, ZIP ou GZ para iniciar.")