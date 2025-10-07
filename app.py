# -*- coding: utf-8 -*-
# Agente de Análise de Dados e Detecção de Fraudes com Gemini SDK (Versão Final Estável)
# Elimina a LangChain para resolver erros de Output Parsing e Depreciação.

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
st.set_page_config(layout="wide", page_title="Multi Agente de Análise Fiscal e de Fraudes (Estável)")

# --- Constantes e Variáveis Globais ---
pio.templates.default = "plotly_white"
MODEL_NAME = "gemini-2.5-flash"

# Tenta obter a chave da API do Gemini do secrets.toml (Streamlit Cloud)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_KEY", "")

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada. Configure a chave no .streamlit/secrets.toml ou na variável de ambiente GEMINI_KEY.")

# --- Inicialização Estável do Gemini ---

@st.cache_resource
def get_gemini_client(api_key, model_name):
    """
    Inicializa e armazena o cliente Gemini na cache para evitar consumo de cota.
    A temperatura (temperature) é definida APENAS na chamada generate_content.
    """
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(
            model_name=model_name
            # 'temperature' é passada na chamada para evitar erros de compatibilidade do SDK
        )
        return client
    except Exception as e:
        st.error(f"Erro fatal ao configurar o Gemini SDK. Verifique sua chave de API. Detalhes: {e}")
        return None

# Chame a função cacheada para obter o cliente
gemini_client = get_gemini_client(API_KEY, MODEL_NAME)

# --- Funções de Manipulação de Arquivos ---

@st.cache_data
def unzip_and_read_file(uploaded_file):
    """
    Descompacta arquivos ZIP ou GZ, lê o conteúdo CSV e retorna o DataFrame
    e o caminho temporário do arquivo (necessário para o contexto do prompt).
    """
    file_extension = uploaded_file.name.lower().split('.')[-1]
    uploaded_file.seek(0)
    
    # Cria o arquivo temporário de forma síncrona
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_csv_path = tmp_file.name

    try:
        # Lógica de descompactação e leitura
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
            os.remove(os.path.abspath(tmp_csv_path)) # Garante a remoção
        return None, None
    
    return tmp_csv_path, df

def get_specialist_prompt(df, temp_csv_path):
    """Gera o prompt de sistema para instruir o Gemini como especialista."""
    # CORREÇÃO CRÍTICA: Enviar APENAS metadados (colunas e tipos) para economizar tokens/tempo
    col_info = df.dtypes.to_markdown()
    
    return f"""
    Você é um Multi Agente de IA SUPER ESPECIALISTA em Contabilidade, Análise de Dados e Desenvolvimento Python.
    Sua missão é analisar dados do arquivo CSV localizado em: {temp_csv_path}.

    **Contexto do Arquivo:**
    - O arquivo possui {df.shape[0]} linhas e {df.shape[1]} colunas.
    - **Tipos de Dados:**
    {col_info}

    **Regras OBRIGATÓRIAS (Essencial para a estabilidade e gráficos):**
    1. **Ferramenta Única:** Você tem acesso à biblioteca Pandas.
    2. **Saída de Gráfico:** **SEMPRE** que o usuário solicitar uma visualização, gere o código Python completo usando **Plotly Express (px)**.
    3. **Formato:** O código Python para o gráfico deve ser **impresso** no formato de string JSON do Plotly, usando o comando:
       `print(f"<PLOTLY_JSON>{{fig.to_json()}}</PLOTLY_JSON>")`
    4. **Caminho do Arquivo:** **SEMPRE** use `pd.read_csv('{temp_csv_path}')` dentro do código Python que você gerar.
    5. **Evitar Quebra:** Mantenha as respostas focadas. Não use raciocínio em etapas ou comandos internos do LangChain que causam erros de parsing.
    6. **Resumo:** Ao final da sua análise ou do código, forneça um resumo claro e conciso da sua conclusão.
    """

def execute_python_code(code_str, temp_csv_path):
    """Executa código Python gerado pelo LLM em um ambiente seguro."""
    # Define o ambiente de execução com o dataframe lido
    exec_globals = {
        'pd': pd,
        'px': px,
        'plt': None, # Remove matplotlib
        'df': pd.read_csv(temp_csv_path),
        'print': print # Permite que o LLM use print para comunicação
    }
    
    # Prepara um buffer para capturar a saída do print
    output_buffer = io.StringIO()
    
    try:
        # Redireciona a saída padrão (stdout) para o nosso buffer
        import sys
        sys.stdout = output_buffer
        
        # Executa o código. O código deve usar 'df'
        exec(code_str, exec_globals)
        
        # Restaura a saída padrão
        sys.stdout = sys.__stdout__
        
        # Retorna a saída capturada (incluindo tags JSON)
        return output_buffer.getvalue()
        
    except Exception as e:
        sys.stdout = sys.__stdout__ # Garantir que o stdout seja restaurado
        return f"ERRO DE EXECUÇÃO PYTHON: {e}\nTraceback: {traceback.format_exc()}"

def parse_and_display_response(response_text):
    """
    Analisa a resposta, extrai e executa código Python e renderiza o gráfico/texto.
    """
    CODE_START = "```python"
    CODE_END = "```"
    PLOTLY_TAG = "<PLOTLY_JSON>"

    if CODE_START in response_text:
        # Separa o texto e o código
        parts = response_text.split(CODE_START, 1)
        text_before = parts[0].strip()
        
        try:
            code_block = parts[1].split(CODE_END, 1)[0].strip()
            text_after = parts[1].split(CODE_END, 1)[1].strip() if len(parts[1].split(CODE_END, 1)) > 1 else ""
        except IndexError:
            # Caso o código Python esteja no final e não tenha o fechamento ```
            code_block = parts[1].strip()
            text_after = ""

        # Executa o código Python
        execution_output = execute_python_code(code_block, st.session_state.temp_csv_path)
        
        # Exibe o texto explicativo antes do código (se houver)
        if text_before:
            st.chat_message("assistant").markdown(text_before)

        # Verifica se o código gerou JSON Plotly (via print)
        if PLOTLY_TAG in execution_output:
            try:
                # Extrai o JSON usando as tags
                json_str = execution_output.split(PLOTLY_TAG, 1)[1].split("</PLOTLY_JSON>", 1)[0].strip()
                fig_dict = json.loads(json_str)
                fig = pio.from_json(json.dumps(fig_dict))
                
                # Renderiza o gráfico
                st.plotly_chart(fig, use_container_width=True)
                
                st.chat_message("assistant").markdown("✅ **Gráfico gerado:** Analise a visualização acima.")
                
            except Exception as e:
                # Se falhar ao decodificar o JSON, exibe o erro
                st.chat_message("assistant").error(f"⚠️ Falha ao renderizar o gráfico. Erro de JSON: {e}")
                
        elif "ERRO DE EXECUÇÃO PYTHON" in execution_output:
            # Exibe erro de execução
            st.chat_message("assistant").error(f"❌ Erro ao executar código Python: {execution_output}")
        
        # Exibe o texto explicativo após o código (se houver)
        if text_after:
            st.chat_message("assistant").markdown(text_after)
            
    else:
        # Se não há código Python, exibe a resposta como texto simples
        st.chat_message("assistant").markdown(response_text)

# --- Layout e Interface ---

st.title("🤖 Multi Agente de Análise Fiscal e de Fraudes")
st.markdown("---")

# Inicialização de estado
if 'temp_csv_path' not in st.session_state:
    st.session_state.temp_csv_path = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history_list' not in st.session_state:
    st.session_state.chat_history_list = []
if 'report_content' not in st.session_state:
    st.session_state.report_content = ""
if 'specialist_prompt' not in st.session_state:
    st.session_state.specialist_prompt = ""
if 'temp_csv_path_to_delete' not in st.session_state:
    st.session_state.temp_csv_path_to_delete = None


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
    st.session_state.temp_csv_path, st.session_state.df = unzip_and_read_file(uploaded_file)
    
    if st.session_state.df is not None:
        st.session_state.specialist_prompt = get_specialist_prompt(st.session_state.df, st.session_state.temp_csv_path)
        st.session_state.chat_history_list.clear() # Limpa o chat ao carregar novo arquivo
        st.success(f"Arquivo '{uploaded_file.name}' carregado e pronto para análise! Pergunte no chat.")
    else:
        st.error("Falha ao carregar o arquivo. Verifique o formato.")

# --- Processamento do Relatório ---

if report_btn and st.session_state.df is not None:
    report_prompt = st.session_state.specialist_prompt + "\n\nFaça uma conclusão resumida e completa de toda a análise de dados realizada até agora, incorporando as seções: Resumo Executivo, Detalhes da Análise, e Conclusão Final. Sua resposta deve ser SOMENTE o conteúdo do relatório em Markdown."
    
    history_context = "\n".join([f"{h['role']}: {h['content']}" for h in st.session_state.chat_history_list])
    
    full_prompt = report_prompt + "\n\nHistórico da Conversa:\n" + history_context

    with st.spinner("Gerando relatório completo..."):
        try:
            response = gemini_client.generate_content(
                full_prompt, 
                config={"temperature": 0.0, "timeout": 180} # Configuração de precisão e timeout
            )
            st.session_state.report_content = response.text
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
        # O parser agora está integrado na função principal
        parse_and_display_response(content)


# Campo de entrada de prompt do usuário
if st.session_state.df is not None and gemini_client:
    if prompt := st.chat_input("Faça sua pergunta ao Agente..."):
        # Adiciona a pergunta ao histórico
        st.session_state.chat_history_list.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.spinner("Agente de IA está processando..."):
            try:
                # Constrói o contexto da conversa
                history_context = "\n".join([f"{item['role']}: {item['content']}" for item in st.session_state.chat_history_list])
                full_context = st.session_state.specialist_prompt + "\n\n" + history_context

                # Chama a API do Gemini com o contexto completo
                response = gemini_client.generate_content(
                    full_context,
                    config={"temperature": 0.0, "timeout": 180} # Configuração de precisão e timeout
                )
                response_text = response.text
                
                # Adiciona a resposta completa ao histórico
                st.session_state.chat_history_list.append({"role": "assistant", "content": response_text})

                # Processa e exibe a resposta (incluindo código/gráfico)
                parse_and_display_response(response_text)

            except Exception as e:
                st.session_state.chat_history_list.append({"role": "assistant", "content": "Ocorreu um erro na comunicação com a IA. Por favor, tente novamente ou reformule sua pergunta."})
                st.chat_message("assistant").error("❌ Erro de comunicação ou timeout. Tente novamente.")
                print(f"Erro na execução da API: {e}")

# Footer
if st.session_state.df is None:
    st.info("⚠️ Carregue um arquivo CSV, ZIP ou GZ para iniciar a análise.")
