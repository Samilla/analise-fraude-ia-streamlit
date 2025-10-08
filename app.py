# -*- coding: utf-8 -*-
# Agente de Análise de Dados e Detecção de Fraudes com Gemini SDK
# Versão com Ferramenta de Diagnóstico de Modelo e Prompt Reforçado

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
st.set_page_config(layout="wide", page_title="Multi Agente de Análise Fiscal (Diagnóstico)")

# --- Constantes e Variáveis Globais ---
pio.templates.default = "plotly_white"
MODEL_NAME = "gemini-2.5-flash"
MAX_HISTORY_SIZE = 8
SAMPLE_ROWS = 100000

# --- Chave de API ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_KEY", "")

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada. Configure-a em .streamlit/secrets.toml.")
    st.stop()

# --- Inicialização e Diagnóstico do Gemini ---
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Erro fatal ao configurar o Gemini SDK com sua chave: {e}")
    st.stop()

@st.cache_resource
def verify_model_availability(model_name):
    """Lista os modelos disponíveis e verifica se o modelo desejado é compatível."""
    try:
        st.write("🔍 Verificando modelos disponíveis para sua chave de API...")
        available_models = [m for m in genai.list_models() if model_name in m.name and 'generateContent' in m.supported_generation_methods]
        
        with st.expander("Clique para ver todos os modelos detectados na sua conta"):
            all_models_info = [{"Nome": m.name, "Métodos Suportados": m.supported_generation_methods} for m in genai.list_models()]
            st.json(all_models_info)

        if available_models:
            st.success(f"✅ Modelo '{available_models[0].name}' encontrado e pronto para uso!")
            return genai.GenerativeModel(model_name=available_models[0].name)
        else:
            st.error(f"❌ ERRO CRÍTICO: O modelo '{model_name}' (ou uma variante) não foi encontrado na sua conta.")
            st.warning("**AÇÕES RECOMENDADAS:**")
            st.markdown("""
                1.  **Verifique se a API 'Vertex AI' está ativada** no seu projeto Google Cloud.
                2.  **Confirme se o Faturamento (Billing)** está habilitado para este projeto.
            """)
            return None
    except Exception as e:
        st.error(f"Falha ao comunicar com a API do Google para listar modelos. Detalhes: {e}")
        return None

gemini_client = verify_model_availability(MODEL_NAME)
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

# --- Lógica do Agente ---

## ALTERAÇÃO 1: Prompt de sistema muito mais rígido e claro.
def get_specialist_prompt(is_sampled, original_rows, col_info):
    """Gera o prompt de sistema para instruir a IA."""
    sampling_info = ""
    if is_sampled:
        sampling_info = (
            f"**ATENÇÃO:** O DataFrame original é muito grande ({original_rows} linhas). "
            f"Para performance, seu código será executado em uma **AMOSTRA ALEATÓRIA de {SAMPLE_ROWS} linhas**. "
            "Sempre mencione em sua análise que os resultados são baseados nesta amostragem."
        )

    return f"""
    Você é um Multi Agente de IA, especialista em Contabilidade, Análise de Dados e Python.

    **REGRAS FUNDAMENTAIS E INQUEBRÁVEIS:**
    1.  **O DataFrame já existe:** Um DataFrame do Pandas já foi carregado e está disponível na variável `df`.
    2.  **NUNCA LEIA ARQUIVOS:** Você está **PROIBIDO** de usar `pd.read_csv()`, `open()`, ou qualquer outra função de leitura de arquivo. Todo o seu código deve operar **DIRETAMENTE** na variável `df`.
    3.  **USE `df` SEMPRE:** Todas as suas operações, análises e gráficos devem usar a variável `df`. Exemplo: `df.describe()`, `px.histogram(df, ...)`.

    **Contexto do DataFrame `df`:**
    - O DataFrame `df` possui {original_rows} linhas no total.
    - {sampling_info}
    - Estrutura das colunas (Tipos de Dados):
    {col_info}

    **REGRAS PARA GERAR GRÁFICOS:**
    1.  **Ferramentas:** Use apenas `pandas as pd` e `plotly.express as px`.
    2.  **SAÍDA OBRIGATÓRIA:** O objeto do gráfico DEVE ser atribuído a uma variável chamada `fig`.
        - Exemplo CORRETO: `fig = px.histogram(df, x='valor_total')`
        - Exemplo INCORRETO: `px.histogram(df, x='valor_total').show()`
    
    Responda textualmente à pergunta do usuário. Se precisar de código para a análise, coloque-o em um bloco ```python ... ```.
    """

def execute_python_code(code_str, temp_csv_path):
    """
    Executa código Python gerado pelo LLM em um ambiente seguro.
    Retorna uma tupla: (output_text, figure_object).
    """
    try:
        df_exec, _, _, _ = get_sampled_df_info(temp_csv_path)
        
        local_scope = {'df': df_exec, 'pd': pd, 'px': px}
        output_buffer = io.StringIO()
        
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_buffer
        
        exec(code_str, {}, local_scope)
        
        sys.stdout = original_stdout
        
        text_output = output_buffer.getvalue()
        fig = local_scope.get('fig', None)
        
        return text_output, fig
        
    except Exception as e:
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
        error_message = f"ERRO DE EXECUÇÃO PYTHON:\n{e}\n\nTraceback:\n{traceback.format_exc()}"
        return error_message, None

## ALTERAÇÃO 2: Lógica de exibição da saída do código aprimorada.
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

    if text_part:
        st.markdown(text_part)

    if code_part:
        with st.expander("Ver Código Executado"):
            st.code(code_part, language='python')
        
        with st.spinner("Executando análise..."):
            text_output, fig_object = execute_python_code(code_part, st.session_state.temp_csv_path)

            # Só exibe a "Saída" se houver algo para mostrar (ignora strings vazias ou só com espaços)
            if text_output.strip() and "ERRO" not in text_output:
                st.info("Saída da Execução:")
                st.text(text_output)
            
            if fig_object:
                st.success("Gráfico gerado com sucesso!")
                st.plotly_chart(fig_object, use_container_width=True)
            
            # Garante que o erro seja exibido claramente
            if "ERRO DE EXECUÇÃO PYTHON" in text_output:
                st.error(text_output)

# --- Layout e Interface ---
st.title("🤖 Multi Agente de Análise Fiscal (com Diagnóstico)")
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
            st.session_state.chat_history = []
            st.success(f"Arquivo '{uploaded_file.name}' carregado. {original_rows} linhas encontradas.")
            st.rerun()
        else:
            st.error("Falha ao carregar o arquivo.")

# --- Interface Principal ---
if not st.session_state.temp_csv_path:
    st.info("⚠️ Por favor, carregue um arquivo na barra lateral para iniciar a análise.")
else:
    for item in st.session_state.chat_history:
        with st.chat_message(item['role']):
            if item['role'] == 'assistant':
                parse_and_display_response(item['content'])
            else:
                st.markdown(item['content'])
    
    if prompt := st.chat_input("Faça sua pergunta ao Agente..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("O Agente está pensando..."):
                try:
                    system_prompt = get_specialist_prompt(**st.session_state.df_metadata)

                    gemini_history = []
                    # Limita o histórico para otimizar a chamada
                    for item in st.session_state.chat_history[-MAX_HISTORY_SIZE:]:
                        role = "user" if item['role'] == 'user' else 'model'
                        # Envia apenas o texto, sem o código, para economizar tokens
                        content_text = item['content'].split("```python")[0].strip()
                        gemini_history.append({'role': role, 'parts': [{'text': content_text}]})
                    
                    if len(gemini_history) > 1 and gemini_history[-2]['role'] == 'user':
                        gemini_history.pop(-2)


                    response = gemini_client.generate_content(
                        gemini_history,
                        generation_config=genai.types.GenerationConfig(temperature=0.0),
                        request_options={'timeout': 300},
                        system_instruction=system_prompt 
                    )
                    
                    full_response_text = response.text
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})
                    
                    parse_and_display_response(full_response_text)

                except Exception as e:
                    error_msg = f"❌ Ocorreu um erro na comunicação com a IA: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    print(f"Erro na execução da API: {e}\n{traceback.format_exc()}")

