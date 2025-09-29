# -*- coding: utf-8 -*-
# Agente de Análise de Dados e Detecção de Fraudes com Gemini e LangChain
# Desenvolvido para um projeto de curso de Agentes de IA.

import streamlit as st
import pandas as pd
import os
import tempfile
import zipfile
import gzip
import io
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory # Usado apenas para o histórico do chat (fora do agente)
from langchain.agents import AgentExecutor
import plotly.express as px
import plotly.io as pio

# --- Configurações Iniciais do Streamlit ---
st.set_page_config(layout="wide", page_title="Multi Agente de Análise de Dados e Fraude")

# --- Constantes e Variáveis Globais ---
pio.templates.default = "plotly_white"
MODEL_NAME = "gemini-2.5-flash"

# Tenta obter a chave da API do Gemini do secrets.toml (Streamlit Cloud)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    # Se estiver rodando localmente sem secrets.toml, usa variável de ambiente
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada. Configure a chave no .streamlit/secrets.toml ou na variável de ambiente GEMINI_API_KEY.")

# --- Funções de Manipulação de Arquivos ---

@st.cache_data(show_spinner="Descompactando arquivo...")
def unzip_and_read_file(uploaded_file):
    """
    Descompacta arquivos ZIP ou GZ e lê o conteúdo CSV.
    Retorna o nome temporário do arquivo e o DataFrame.
    """
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    if file_extension == 'zip':
        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            # Tenta encontrar o primeiro CSV dentro do ZIP
            csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
            if not csv_files:
                return None, None
            
            # Lê o primeiro CSV encontrado
            with zf.open(csv_files[0]) as csv_file:
                df = pd.read_csv(csv_file)
                # Salva o arquivo em disco temporário para o agente acessar
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    df.to_csv(tmp_file.name, index=False)
                    return tmp_file.name, df
    
    elif uploaded_file.name.endswith('.gz'):
        # Lida com arquivos GZ (que podem ser CSVs gzipados)
        try:
            with gzip.open(uploaded_file, 'rt') as gz_file:
                df = pd.read_csv(gz_file)
                # Salva o arquivo em disco temporário para o agente acessar
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    df.to_csv(tmp_file.name, index=False)
                    return tmp_file.name, df
        except Exception:
            return None, None

    elif file_extension == 'csv':
        # Arquivo CSV normal
        df = pd.read_csv(uploaded_file)
        # Salva o arquivo em disco temporário para o agente acessar
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            return tmp_file.name, df
    
    return None, None

# --- Funções do Agente ---

def load_llm_and_memory(temp_csv_path):
    """
    Cria e carrega a memória e o agente de IA com o prompt de especialista.
    """
    # 1. Prompt do Especialista (Regras Rígidas)
    analyst_prompt = f"""
    Você é um Multi Agente de IA SUPER ESPECIALISTA em Contabilidade, Análise de Dados e Desenvolvimento Python.
    Sua missão é atuar como um Analista de Dados e Fraudes, capaz de analisar qualquer arquivo CSV fiscal, contábil ou operacional fornecido pelo usuário.

    **Você DEVE seguir estas regras estritamente:**
    1. **Personalidade:** Seja objetivo, técnico e focado em fornecer insights e código Python (quando solicitado).
    2. **Foco:** Use as colunas e dados do arquivo CSV fornecido, que está em '{temp_csv_path}', para responder a todas as perguntas.
    3. **Memória:** Use o histórico de chat fornecido para manter o contexto e as conclusões.
    4. **Gráficos:** **SEMPRE** que o usuário solicitar uma visualização, utilize a biblioteca **Plotly**.
       **Atenção:** **NUNCA USE MATPLOTLIB OU SEABORN.**
       O seu output final para gráficos **DEVE** ser uma string JSON válida do Plotly (`fig.to_json()`) para que o Streamlit possa renderizar a imagem.
    5. **Saída Final:** O resultado final de sua análise deve ser claro e conciso.

    **Instruções Específicas do Usuário:** O usuário forneceu instruções específicas no sidebar que você deve integrar à sua análise: "{st.session_state.user_instructions}"
    """

    # 2. Configuração do Modelo e Agente
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY)
    except Exception as e:
        # Erro na inicialização do LLM
        st.error(f"Erro fatal ao inicializar o LLM Gemini. Detalhes: {e}")
        return None, None 
    
    # 3. Inicialização da Memória (fora do agente)
    # A memória será gerenciada manualmente via histórico do chat (chat_history_list)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="input",
        return_messages=True,
        k=5, 
        ai_prefix="Analista"
    )

    # 4. Criação do Agente (Bloco de segurança final)
    try:
        # A memória foi removida daqui para evitar o conflito de inicialização
        agent = create_csv_agent(
            llm=llm,
            path=temp_csv_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            extra_tools=None,
            prefix=analyst_prompt,
            allow_dangerous_code=True
        )
        return memory, agent
    except Exception as e:
        # Este é o erro que deve ser exposto se a API falhar
        print(f"DEBUG: FALHA CRÍTICA NA CRIAÇÃO DO AGENTE (PROVAVELMENTE CONEXÃO OU VERSÃO): {e}")
        st.error(f"Erro CRÍTICO ao criar o Agente CSV. Verifique a API Key e as logs. Detalhes: {e}")
        return None, None 

def parse_and_display_response(response_text):
    """
    Analisa a resposta do agente, procurando por JSON do Plotly.
    Se encontrar, renderiza o gráfico; caso contrário, exibe como texto.
    """
    try:
        # Padrão para tentar extrair um JSON Plotly
        # O agente deve retornar o JSON como uma string, muitas vezes entre aspas ou chaves
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = response_text[start_index:end_index + 1]
            # Tenta carregar o JSON e renderizar o gráfico
            fig_dict = json.loads(json_str)
            fig = pio.from_json(json.dumps(fig_dict))
            
            # Renderiza o gráfico e exibe a mensagem de sucesso
            st.plotly_chart(fig, use_container_width=True)
            
            # Remove o JSON da resposta para exibir apenas o texto explicativo
            text_response = response_text[:start_index].strip()
            if not text_response:
                text_response = "Gráfico gerado com sucesso. Analise a visualização acima."
                
            return text_response
    
    except Exception as e:
        # Se falhar ao processar o JSON, assume que é apenas texto
        pass

    # Se não for JSON Plotly, retorna a resposta como está
    return response_text

# --- Layout e Interface ---

st.title("🤖 Multi Agente de Análise Fiscal e de Fraudes")
st.markdown("---")

# Inicialização de estado para memória e agente
if 'data_agent' not in st.session_state:
    st.session_state.data_agent = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'chat_history_list' not in st.session_state:
    st.session_state.chat_history_list = []
if 'report_content' not in st.session_state:
    st.session_state.report_content = ""
if 'user_instructions' not in st.session_state:
    st.session_state.user_instructions = "Nenhuma instrução específica fornecida."

# --- Barra Lateral (Configurações e Upload) ---

with st.sidebar:
    st.header("⚙️ Configurações de Análise")
    
    # 1. Upload de Arquivo
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV, ZIP ou GZ:",
        type=['csv', 'zip', 'gz'],
        key="file_uploader"
    )
    
    # 2. Instruções Personalizadas
    st.subheader("Instruções de Análise")
    instructions = st.text_area(
        "Instruções e Foco do Agente (Ex: Focar em contas de alto risco):",
        height=150,
        placeholder="Nenhuma instrução específica. (O agente fará uma análise geral)",
        key="user_instructions_input"
    )
    st.session_state.user_instructions = instructions if instructions else "Nenhuma instrução específica fornecida."
    
    # 3. Botão de Relatório Completo
    st.subheader("Relatório Final")
    report_btn = st.button("📝 Gerar Relatório Completo", use_container_width=True)
    
    # 4. Botão de Download (Depende do conteúdo do relatório)
    if st.session_state.report_content:
        st.download_button(
            label="⬇️ Baixar Relatório (Markdown)",
            data=st.session_state.report_content,
            file_name="relatorio_analise_ia.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- Processamento do Arquivo ---

if uploaded_file and st.session_state.data_agent is None:
    temp_csv_path, df = unzip_and_read_file(uploaded_file)
    
    if temp_csv_path and df is not None:
        try:
            # Inicializa o agente
            st.session_state.memory, st.session_state.data_agent = load_llm_and_memory(temp_csv_path)
            
            # Adiciona mensagem de sucesso
            if st.session_state.data_agent is not None:
                st.success(f"Arquivo '{uploaded_file.name}' carregado e agente inicializado!")
                
                # Pergunta inicial para auto-descrição do arquivo
                initial_q1 = f"Descreva os dados: Quais são as colunas, tipos de dados e o formato geral do arquivo? {df.shape[0]} linhas e {df.shape[1]} colunas."
                
                # Executa a primeira análise para preencher o chat
                with st.spinner(f"Agente analisando o arquivo..."):
                    try:
                        # Executa a pergunta inicial e armazena a resposta
                        response_obj = st.session_state.data_agent.run(initial_q1)
                        initial_response = parse_and_display_response(response_obj)
                        
                        st.session_state.chat_history_list.append(("user", initial_q1))
                        st.session_state.chat_history_list.append(("agent", initial_response))
                        
                    except Exception as e:
                        # Lida com erros de parsing logo na primeira pergunta
                        st.session_state.chat_history_list.append(("user", initial_q1))
                        st.session_state.chat_history_list.append(("agent", f"O Agente inicializou, mas encontrou um erro ao tentar a primeira análise. O problema de 'Output Parsing' pode ocorrer. Por favor, faça uma pergunta simples como 'Quais colunas existem?' para testar o agente."))
                        print(f"Erro no primeiro parsing: {e}")
            else:
                st.error("Falha ao inicializar o agente. Verifique a chave da API e as mensagens de erro na lateral.")
                    
        except Exception as e:
            st.error(f"Erro grave ao processar o arquivo: {e}")
    
    elif uploaded_file:
        st.error("Formato de arquivo não suportado ou erro ao descompactar. Certifique-se de que é um CSV válido, ZIP ou GZ.")

# --- Processamento do Relatório ---

if report_btn and st.session_state.data_agent:
    # Pergunta que força o agente a gerar um relatório completo
    report_prompt = """
    Gere um relatório completo da análise de dados realizada até agora, incorporando as seguintes seções em Markdown formatado:
    1. **Resumo Executivo (Conclusão do Agente):** Quais são as principais conclusões encontradas e padrões/anomalias mais importantes?
    2. **Descrição dos Dados:** Detalhes de colunas, tipos de dados, e medidas centrais/dispersão.
    3. **Relações e Tendências:** Quais foram as correlações mais fortes e tendências identificadas.
    4. **Sugestões:** Recomendações finais baseadas na análise (Ex: colunas a investigar para fraude).

    Sua resposta deve ser SOMENTE o conteúdo do relatório em Markdown.
    """
    
    with st.spinner("Gerando relatório completo... Isso pode levar alguns momentos."):
        try:
            # Roda o agente com o prompt de relatório
            report_response = st.session_state.data_agent.run(report_prompt)
            
            st.session_state.report_content = report_response
            st.success("Relatório gerado com sucesso! Use o botão 'Baixar Relatório (Markdown)' na lateral.")
            
        except Exception as e:
            st.error(f"Erro ao gerar o relatório: {e}")

# --- Interface de Chat ---

# Exibe o histórico de chat
for role, message in st.session_state.chat_history_list:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        # Usa o parser para renderizar a mensagem do agente (incluindo gráficos)
        parsed_message = parse_and_display_response(message)
        st.chat_message("assistant").markdown(parsed_message)

# Campo de entrada de prompt do usuário
if st.session_state.data_agent:
    if prompt := st.chat_input("Faça sua pergunta ao Agente... (Ex: Qual a correlação entre Amount e Time?)"):
        st.session_state.chat_history_list.append(("user", prompt))
        st.chat_message("user").markdown(prompt)

        with st.spinner("Agente de IA está processando..."):
            try:
                # Executa a pergunta e armazena a resposta
                # Usando .run() para maior simplicidade de saída, mas com parsing robusto
                response_obj = st.session_state.data_agent.run(prompt)
                
                # Garante que a resposta completa seja armazenada para análise e memória
                parsed_response = parse_and_display_response(response_obj)
                
                # Adiciona a resposta ao histórico
                st.session_state.chat_history_list.append(("agent", response_obj))
                
                # Exibe a resposta final (se não for um gráfico, será exibido como texto)
                st.chat_message("assistant").markdown(parsed_response)

            except Exception as e:
                # Tratamento robusto de erro para evitar quebras do aplicativo
                st.session_state.chat_history_list.append(("agent", "O Agente encontrou um erro ao processar sua requisição. Por favor, tente reformular a pergunta ou verificar se o arquivo CSV está bem formatado. Detalhes: An output parsing error occurred. Tente reiniciar o aplicativo se o erro persistir."))
                print(f"Erro na execução do agente (run): {e}")
                st.experimental_rerun() # Reinicia para limpar o estado em caso de erro grave

# Footer para indicar o estado
if st.session_state.data_agent is None:
    st.info("⚠️ Carregue um arquivo CSV, ZIP ou GZ para iniciar a análise.")

# Limpa o arquivo temporário ao finalizar o Streamlit
if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
    os.remove(temp_csv_path)

