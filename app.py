# -*- coding: utf-8 -*-
# Agente de Análise de Dados e Detecção de Fraudes com Gemini e LangChain
# Desenvolvido para um projeto de curso de Agentes de IA.
# Versão Final Corrigida e Estável.

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
from langchain.memory import ConversationBufferWindowMemory # Usando WindowMemory para estabilidade e contexto
import plotly.express as px
import plotly.io as pio

# --- Configurações Iniciais do Streamlit ---
st.set_page_config(layout="wide", page_title="Multi Agente de Análise Fiscal e de Fraudes")

# --- Constantes e Variáveis Globais ---
pio.templates.default = "plotly_white"
# Modelo final e estável
MODEL_NAME = "gemini-2.5-flash"

# Tenta obter a chave da API do Gemini do secrets.toml (Streamlit Cloud)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    # Se estiver rodando localmente sem secrets.toml, usa variável de ambiente
    API_KEY = os.environ.get("GEMINI_KEY", "") # Nota: 'GEMINI_KEY' é comum em ambientes

if not API_KEY:
    st.error("ERRO: Chave da API do Gemini não encontrada. Configure a chave no .streamlit/secrets.toml ou na variável de ambiente GEMINI_KEY.")

# --- Funções de Manipulação de Arquivos ---

def unzip_and_read_file(uploaded_file):
    """
    Descompacta arquivos ZIP ou GZ e lê o conteúdo CSV.
    Retorna o nome temporário do arquivo e o DataFrame.
    """
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    # Resetar o ponteiro do arquivo
    uploaded_file.seek(0)
    
    # Cria o arquivo temporário de forma síncrona
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_csv_path = tmp_file.name

    try:
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
        # Limpa o arquivo temporário se houve falha
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
        return None, None
    
    return None, None

# --- Funções do Agente ---

@st.cache_resource
def load_llm_and_memory(temp_csv_path):
    """
    Cria e carrega a memória e o agente de IA com o prompt de especialista.
    **CACHED:** Esta função é executada apenas uma vez por sessão.
    """
    # 1. Prompt do Especialista (Regras Rígidas)
    analyst_prompt = f"""
    Você é um Multi Agente de IA SUPER ESPECIALISTA em Contabilidade, Análise de Dados e Desenvolvimento Python.
    Sua missão é atuar como um Analista de Dados e Fraudes, capaz de analisar qualquer arquivo CSV fiscal, contábil ou operacional fornecido pelo usuário.

    **Você DEVE seguir estas regras estritamente:**
    1. **Personalidade:** Seja objetivo, técnico e focado em fornecer insights e código Python (quando solicitado).
    2. **Foco:** Use as colunas e dados do arquivo CSV fornecido, que está em '{temp_csv_path}', para responder a todas as perguntas.
    3. **Gráficos (CRÍTICO):** **SEMPRE** que o usuário solicitar uma visualização, utilize a biblioteca **Plotly**.
       **Atenção:** **NUNCA USE MATPLOTLIB OU SEABORN.**
       O seu output final para gráficos **DEVE** ser uma string JSON válida do Plotly (`fig.to_json()`), **delimitada OBRIGATORIAMENTE pelas tags <PLOTLY_JSON> e </PLOTLY_JSON>**. Nenhuma outra informação deve estar dentro dessas tags.
    4. **Saída Final:** O resultado final de sua análise deve ser claro e conciso.
    5. **Ação Final:** O Agente deve responder APENAS com o resultado da sua análise. NUNCA use tags de 'Final Answer'.
    """

    # 2. Configuração do Modelo e Agente
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            google_api_key=API_KEY,
            # Adicionando temperatura baixa e timeout para estabilidade
            temperature=0.0, 
            timeout=120  # Aumenta o tempo limite para 120 segundos
        )
    except Exception as e:
        st.error(f"Erro fatal ao inicializar o LLM Gemini. Detalhes: {e}")
        return None, None 
    
    # 3. Inicialização da Memória (Memória base para evitar warnings)
    # Correção FINAL do Warning: Estrutura explícita de input/output
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",  # Adicionado para evitar incompatibilidade interna
        return_messages=True,
        ai_prefix="Analista",
        k=5 
    )

    # 4. Criação do Agente (Bloco de segurança final)
    try:
        # Usando OPENAI_FUNCTIONS para estabilidade de parsing com Gemini
        agent_executor = create_csv_agent(
            llm=llm,
            path=temp_csv_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS, 
            prefix=analyst_prompt,
            allow_dangerous_code=True
        )
        return memory, agent_executor
    except Exception as e:
        # Este é o erro que deve ser exposto se a API falhar
        print(f"DEBUG: FALHA CRÍTICA NA CRIAÇÃO DO AGENTE (PROVAVELMENTE CONEXÃO OU VERSÃO): {e}")
        st.error(f"Erro CRÍTICO ao criar o Agente CSV. Verifique a API Key e as logs. Detalhes: {e}")
        return None, None 

def parse_and_display_response(response_text):
    """
    Analisa a resposta do agente, procurando por JSON do Plotly usando as tags
    <PLOTLY_JSON>...</PLOTLY_JSON>. Se encontrar, renderiza o gráfico; caso contrário,
    exibe como texto.
    """
    START_TAG = "<PLOTLY_JSON>"
    END_TAG = "</PLOTLY_JSON>"

    start_index = response_text.find(START_TAG)
    end_index = response_text.find(END_TAG)
    
    if start_index != -1 and end_index != -1 and end_index > start_index:
        # Extrai o JSON que está entre as tags
        json_str = response_text[start_index + len(START_TAG):end_index].strip()
        
        try:
            # Tenta carregar o JSON e renderizar o gráfico
            fig_dict = json.loads(json_str)
            fig = pio.from_json(json.dumps(fig_dict))
            
            # Renderiza o gráfico
            st.plotly_chart(fig, use_container_width=True)
            
            # Remove o JSON e as tags da resposta para exibir apenas o texto explicativo
            text_response = response_text.replace(response_text[start_index:end_index + len(END_TAG)], "").strip()
            if not text_response:
                text_response = "Gráfico gerado com sucesso. Analise a visualização acima."
                
            return text_response
    
        except Exception as e:
            # Se falhar ao processar o JSON extraído, loga o erro e trata como texto
            print(f"DEBUG: Falha na decodificação do JSON Plotly extraído. Erro: {e}")
            pass

    # Se não encontrou as tags ou falhou na decodificação, retorna o texto original
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
if 'temp_csv_path' not in st.session_state:
    st.session_state.temp_csv_path = None

# --- Barra Lateral (Configurações e Upload) ---

with st.sidebar:
    st.header("⚙️ Configurações de Análise")
    
    # 1. Upload de Arquivo
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV, ZIP ou GZ:",
        type=['csv', 'zip', 'gz'],
        key="file_uploader"
    )
    
    # REMOVIDA: Seção de Instruções Personalizadas
    
    # 2. Botão de Relatório Completo
    st.subheader("Relatório Final")
    report_btn = st.button("📝 Gerar Relatório Completo", use_container_width=True)
    
    # 3. Botão de Download (Depende do conteúdo do relatório)
    if st.session_state.report_content:
        st.download_button(
            label="⬇️ Baixar Relatório (Markdown)",
            data=st.session_state.report_content,
            file_name="relatorio_analise_ia.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- Processamento do Arquivo ---

# Verifica se um novo arquivo foi carregado ou se é a primeira execução
if uploaded_file and st.session_state.data_agent is None:
    temp_csv_path, df = unzip_and_read_file(uploaded_file)
    st.session_state.temp_csv_path = temp_csv_path
    
    if st.session_state.temp_csv_path and df is not None:
        try:
            # Inicializa o agente
            st.session_state.memory, st.session_state.data_agent = load_llm_and_memory(st.session_state.temp_csv_path)
            
            # Adiciona mensagem de sucesso
            if st.session_state.data_agent is not None:
                # MENSAGEM DE SUCESSO AJUSTADA
                st.success(f"Arquivo '{uploaded_file.name}' carregado e agente inicializado! Faça sua primeira pergunta no chat abaixo, como 'Quais colunas existem?'")
                
                # REMOVIDO: Bloco de execução da pergunta inicial (initial_q1)
                # O usuário fará a primeira pergunta diretamente para evitar falhas de parsing.
                        
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
            # Usando .invoke() em vez de .run() (Deprecation fix)
            report_response = st.session_state.data_agent.invoke({"input": report_prompt})['output']
            
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
                # Usando .invoke() em vez de .run() (Deprecation fix)
                response_obj = st.session_state.data_agent.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.memory.load_memory_variables({})['chat_history']
                    }
                )['output']
                
                # Garante que a resposta completa seja armazenada para análise e memória
                parsed_response = parse_and_display_response(response_obj)
                
                # Adiciona a resposta ao histórico (para display e memória)
                st.session_state.chat_history_list.append(("agent", response_obj))
                
                # Exibe a resposta final (se não for um gráfico, será exibido como texto)
                st.chat_message("assistant").markdown(parsed_response)

                # Atualiza a memória para incluir a última interação (user + agent)
                st.session_state.memory.save_context(
                    {"input": prompt},
                    {"output": response_obj}
                )

            except Exception as e:
                # Tratamento robusto de erro para evitar quebras do aplicativo
                st.session_state.chat_history_list.append(("agent", "O Agente encontrou um erro ao processar sua requisição. Por favor, tente reformular a pergunta ou verificar se o arquivo CSV está bem formatado."))
                print(f"Erro na execução do agente (run): {e}")

# Footer para indicar o estado
if st.session_state.data_agent is None:
    st.info("⚠️ Carregue um arquivo CSV, ZIP ou GZ para iniciar a análise.")

# Limpa o arquivo temporário ao finalizar o Streamlit
def cleanup_temp_file():
    if st.session_state.temp_csv_path and os.path.exists(st.session_state.temp_csv_path):
        os.remove(st.session_state.temp_csv_path)

# Adiciona a função de limpeza na finalização da sessão
# Embora Streamlit não tenha um hook de finalização de sessão garantido, isso ajuda.
# O sistema operacional se encarrega da limpeza dos arquivos temporários em caso de falha.