# -*- coding: utf-8 -*-
# Agente de Análise de Dados e Detecção de Fraudes com Gemini SDK
# Versão Refatorada: Gráficos Robustos e Gerenciamento de Histórico Otimizado

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

# --- Configurações Iniciais do Streamlit ---
st.set_page_config(layout="wide", page_title="Multi Agente de Análise Fiscal e de Fraudes (Otimizado)")

# --- Constantes e Variáveis Globais ---
pio.templates.default = "plotly_white"
MODEL_NAME = "gemini-1.5-flash" ## ALTERAÇÃO: Usando o Gemini 1.5 Flash para melhor capacidade de seguir instruções.
MAX_HISTORY_SIZE = 8 # Reduzido um pouco, pois o contexto é mais limpo agora.
SAMPLE_ROWS = 100000

# Tenta obter a chave da API do Gemini do secrets.toml (Streamlit Cloud)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_KEY", "")

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada. Configure-a em .streamlit/secrets.toml.")
    st.stop()

# --- Inicialização Estável do Gemini ---
@st.cache_resource
def get_gemini_client(api_key, model_name):
    """Inicializa e armazena o cliente Gemini na cache."""
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model_name=model_name)
        return client
    except Exception as e:
        st.error(f"Erro fatal ao configurar o Gemini SDK: {e}")
        return None

gemini_client = get_gemini_client(API_KEY, MODEL_NAME)
if not gemini_client:
    st.stop()

# --- Funções de Manipulação de Arquivos (Sem alterações) ---
@st.cache_data
def unzip_and_read_file(uploaded_file):
    """Descompacta arquivos, lê o CSV e retorna o DataFrame e o caminho temporário."""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    uploaded_file.seek(0)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_csv_path = tmp_file.name

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
        elif file_extension in ['gz', 'gzip']:
            with gzip.open(uploaded_file, 'rt', encoding='utf-8') as gz_file:
                df = pd.read_csv(gz_file)
                df.to_csv(tmp_csv_path, index=False)
                return tmp_csv_path, df
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            df.to_csv(tmp_csv_path, index=False)
            return tmp_csv_path, df
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None, None
    return None, None

@st.cache_data(show_spinner="Analisando e cacheando DataFrame...")
def get_sampled_df_info(temp_csv_path):
    """Lê o DataFrame, aplica amostragem se necessário e retorna metadados."""
    df = pd.read_csv(temp_csv_path)
    original_rows = len(df)
    is_sampled = False
    if original_rows > SAMPLE_ROWS:
        df_sampled = df.sample(n=SAMPLE_ROWS, random_state=42)
        is_sampled = True
        return df_sampled, is_sampled, original_rows, df.dtypes.to_markdown()
    return df, is_sampled, original_rows, df.dtypes.to_markdown()

# --- Lógica do Agente (Grandes Alterações Aqui) ---

def get_specialist_prompt(is_sampled, original_rows, col_info):
    """
    ## ALTERAÇÃO: Prompt simplificado e mais direto.
    Instrui a IA a gerar código que atribui o gráfico a uma variável `fig`.
    """
    sampling_info = ""
    if is_sampled:
        sampling_info = (
            f"**ATENÇÃO:** O DataFrame original é muito grande ({original_rows} linhas). "
            f"Para performance, todo o código que você gerar será executado em uma **AMOSTRA ALEATÓRIA de {SAMPLE_ROWS} linhas**. "
            "Sempre mencione em sua análise que os resultados são baseados nesta amostragem."
        )

    return f"""
    Você é um Multi Agente de IA, especialista em Contabilidade, Análise de Dados e Python.
    Sua missão é responder perguntas sobre um DataFrame do Pandas que já está carregado na memória como `df`.

    **Contexto do DataFrame:**
    - O DataFrame `df` possui {original_rows} linhas no total.
    - {sampling_info}
    - Estrutura das colunas (Tipos de Dados):
    {col_info}

    **REGRAS CRÍTICAS PARA GERAR GRÁFICOS:**
    1.  **Ferramentas:** Use apenas as bibliotecas `pandas as pd` e `plotly.express as px`.
    2.  **Execução:** Você vai gerar blocos de código Python para análise. Este código será executado em um ambiente seguro.
    3.  **Visualização:** Se o usuário pedir um gráfico, gere o código Python usando Plotly Express.
    4.  **SAÍDA OBRIGATÓRIA PARA GRÁFICOS:** O objeto do gráfico DEVE ser atribuído a uma variável chamada `fig`.
        - Exemplo CORRETO: `fig = px.histogram(df, x='valor_total')`
        - Exemplo INCORRETO: `px.histogram(df, x='valor_total').show()`
    5.  **Análise e Código:** Responda textualmente à pergunta do usuário. Se precisar de código, coloque-o em um bloco ```python ... ```.
    6.  **Clareza:** Seja direto. Primeiro o texto da análise, depois o bloco de código se necessário.
    """

## ALTERAÇÃO: Função de execução totalmente refeita para ser mais robusta.
def execute_python_code(code_str, temp_csv_path):
    """
    Executa código Python gerado pelo LLM em um ambiente seguro.
    Retorna uma tupla: (output_text, figure_object).
    """
    try:
        df_exec, _, _, _ = get_sampled_df_info(temp_csv_path)
        
        local_scope = {'df': df_exec, 'pd': pd, 'px': px}
        output_buffer = io.StringIO()
        
        # Redireciona stdout para capturar 'print'
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_buffer
        
        # Executa o código no escopo local
        exec(code_str, {}, local_scope)
        
        # Restaura stdout
        sys.stdout = original_stdout
        
        text_output = output_buffer.getvalue()
        
        # Procura pela variável 'fig' no escopo local
        fig = local_scope.get('fig', None)
        
        return text_output, fig
        
    except Exception as e:
        # Restaura stdout em caso de erro também
        sys.stdout = original_stdout
        error_message = f"ERRO DE EXECUÇÃO PYTHON:\n{e}\n\nTraceback:\n{traceback.format_exc()}"
        return error_message, None

## ALTERAÇÃO: Função de parsing completamente redesenhada.
def parse_and_display_response(full_response_text):
    """
    Analisa a resposta, extrai e executa código Python e renderiza o gráfico/texto.
    """
    text_part = full_response_text
    code_part = None
    
    if "```python" in full_response_text:
        parts = full_response_text.split("```python", 1)
        text_part = parts[0].strip()
        code_part = parts[1].split("```", 1)[0].strip()

    # Exibe a parte textual da resposta da IA
    if text_part:
        st.markdown(text_part)

    # Se houver código, executa e exibe os resultados
    if code_part:
        with st.expander("Ver Código Executado"):
            st.code(code_part, language='python')
        
        with st.spinner("Executando análise..."):
            text_output, fig_object = execute_python_code(code_part, st.session_state.temp_csv_path)

            if text_output:
                st.info("Saída da Execução:")
                st.text(text_output)
            
            if fig_object:
                st.success("Gráfico gerado com sucesso!")
                st.plotly_chart(fig_object, use_container_width=True)
            elif "ERRO" in text_output:
                st.error(text_output)

# --- Layout e Interface ---
st.title("🤖 Multi Agente de Análise Fiscal e de Fraudes (Otimizado)")
st.markdown("---")

# Inicialização de estado
if 'temp_csv_path' not in st.session_state:
    st.session_state.temp_csv_path = None
if 'df_metadata' not in st.session_state:
    st.session_state.df_metadata = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Barra Lateral ---
with st.sidebar:
    st.header("⚙️ 1. Configuração da Análise")
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo (CSV, ZIP, GZ):",
        type=['csv', 'zip', 'gz'],
        key="file_uploader"
    )

# --- Processamento do Arquivo ---
if uploaded_file and not st.session_state.temp_csv_path:
    with st.spinner("Processando e analisando o arquivo..."):
        temp_path, _ = unzip_and_read_file(uploaded_file)
        if temp_path:
            st.session_state.temp_csv_path = temp_path
            _, is_sampled, original_rows, col_info = get_sampled_df_info(temp_path)
            st.session_state.df_metadata = {
                "is_sampled": is_sampled,
                "original_rows": original_rows,
                "col_info": col_info
            }
            st.session_state.chat_history = [] # Limpa chat
            st.success(f"Arquivo '{uploaded_file.name}' carregado. {original_rows} linhas encontradas.")
            st.rerun() # Força o recarregamento para atualizar a UI
        else:
            st.error("Falha ao carregar o arquivo.")

# --- Interface Principal ---
if not st.session_state.temp_csv_path:
    st.info("⚠️ Por favor, carregue um arquivo na barra lateral para iniciar a análise.")
else:
    # Exibe o histórico de chat
    for item in st.session_state.chat_history:
        with st.chat_message(item['role']):
            # A função de display agora lida com tudo
            if item['role'] == 'assistant':
                parse_and_display_response(item['content'])
            else:
                st.markdown(item['content'])
    
    # Campo de entrada de prompt do usuário
    if prompt := st.chat_input("Faça sua pergunta ao Agente..."):
        # Adiciona pergunta do usuário ao histórico e à UI
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("O Agente está pensando..."):
                try:
                    # Prepara o prompt de sistema
                    system_prompt = get_specialist_prompt(**st.session_state.df_metadata)

                    # ## ALTERAÇÃO: Construção de histórico otimizada
                    # Converte o histórico para o formato do Gemini, omitindo código
                    gemini_history = []
                    for item in st.session_state.chat_history[-MAX_HISTORY_SIZE:]:
                        role = "user" if item['role'] == 'user' else 'model'
                        # Apenas a parte textual entra no histórico para economizar tokens
                        content_text = item['content'].split("```python")[0].strip()
                        gemini_history.append({'role': role, 'parts': [{'text': content_text}]})
                    
                    # Remove a última mensagem (que é a do usuário atual, que já está em 'prompt')
                    gemini_history.pop() 

                    # Cria o cliente de chat com histórico e instrução de sistema
                    chat = gemini_client.start_chat(history=gemini_history)
                    
                    response = chat.send_message(
                         prompt,
                         generation_config=genai.types.GenerationConfig(temperature=0.0),
                         stream=False, # Streaming é mais complexo com execução de código, vamos simplificar por enquanto
                         request_options={'timeout': 300}
                    )
                    
                    full_response_text = response.text
                    
                    # Adiciona a resposta completa ao histórico para ser parseada
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})
                    
                    # A resposta é renderizada aqui, dentro da mensagem de chat
                    parse_and_display_response(full_response_text)

                except Exception as e:
                    error_msg = f"❌ Ocorreu um erro na comunicação com a IA: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    print(f"Erro na execução da API: {e}\n{traceback.format_exc()}")
