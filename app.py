import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import plotly.express as px
import plotly.io as pio # Necessário para ler o JSON do Plotly
import tempfile
import os
import logging
import zipfile
import gzip
import io
import json # Necessário para parsing de JSON
from langchain.agents import AgentExecutor

# Configuração de Logs para Debug (Útil no ambiente local)
logging.basicConfig(level=logging.INFO)

# --- Configuração da Página e da API Key ---
st.set_page_config(
    page_title="Multi Agente Analista de CSV Fiscal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verifica a chave de API
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Chave de API do Gemini não encontrada. Configure GEMINI_API_KEY no secrets.toml.")
    st.stop()

# --- Funções de Inicialização e Processamento de Arquivos ---

def load_llm_and_memory(custom_instructions=""):
    """Carrega o LLM e configura a memória de conversa com instruções personalizadas."""
    try:
        # Inicializa o LLM (Gemini Flash é ideal para esta tarefa)
        llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=st.secrets["GEMINI_API_KEY"],
            temperature=0.1
        )
        
        # Configura a memória do agente
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # --- PROMPT GENERALIZADO COM FOCO FISCAL E INSTRUÇÕES DINÂMICAS ---
        analyst_prompt = f"""
        Você é um **Multi Agente de IA SUPER ESPECIALISTA** em Contabilidade, Big Data, Análise de Dados e Python.
        Sua principal tarefa é realizar a **Análise Exploratória de Dados (EDA)** detalhada para fins fiscais, contábeis ou de detecção de fraudes.

        **Siga estas regras:**
        1. **Foco Fiscal/Contábil:** Suas análises devem procurar por anomalias, padrões e tendências relevantes para balanços, DRE, detecção de gastos irregulares, ou análise de transações.
        2. **Instruções do Usuário:** Use as regras fornecidas pelo usuário ({custom_instructions}) para guiar suas análises e decisões.
        3. **Ferramentas:** Use a ferramenta CSV Agent para gerar código Python, realizar cálculos estatísticos e extrair informações.
        4. **Gráficos:** **SEMPRE** que o usuário solicitar uma visualização, utilize **Plotly**. **NUNCA USE MATPLOTLIB OU SEABORN**. Quando gerar um gráfico, o código Python da ferramenta DEVE **imprimir a representação JSON do gráfico** usando `fig.to_json()` para que o Streamlit possa renderizá-lo. Após gerar o JSON, **não inclua mais texto na resposta**.
        5. **Linguagem:** Responda **sempre em Português**, de forma clara e profissional.
        6. **Conclusões Finais:** Use a memória de chat e as análises para gerar uma seção de conclusões detalhadas sobre o conjunto de dados.
        
        Histórico da Conversa: {{chat_history}}
        Pergunta do Usuário: {{input}}
        Resposta:
        """
        # A ConversationChain é usada para a memória da sessão e o relatório final.
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=PromptTemplate(template=analyst_prompt, input_variables=["chat_history", "input"])
        )

        return llm, conversation, memory
    
    except Exception as e:
        st.error(f"Erro ao inicializar o LLM: {e}")
        return None, None, None

def decompress_file(uploaded_file):
    """Descompacta arquivos ZIP ou GZ e retorna o caminho para o CSV temporário."""
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    
    # 1. Cria um arquivo temporário no disco para o arquivo CSV final
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        tmp_csv_path = tmp_csv.name
    
    # 2. Processa arquivos ZIP
    if file_extension == 'zip':
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()), 'r') as zip_ref:
                # Procura pelo primeiro arquivo CSV dentro do ZIP
                csv_in_zip = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
                with zip_ref.open(csv_in_zip) as csv_file:
                    with open(tmp_csv_path, 'wb') as f:
                        f.write(csv_file.read())
                st.sidebar.success(f"Arquivo ZIP descompactado. Analisando '{csv_in_zip}'.")
                return tmp_csv_path
        except Exception as e:
            st.sidebar.error(f"Erro ao descompactar o arquivo ZIP: {e}")
            return None
    
    # 3. Processa arquivos GZ (GZIP)
    elif file_extension == 'gz':
        try:
            with gzip.open(uploaded_file, 'rb') as f_in:
                with open(tmp_csv_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            st.sidebar.success(f"Arquivo GZ descompactado. Analisando o conteúdo.")
            return tmp_csv_path
        except Exception as e:
            st.sidebar.error(f"Erro ao descompactar o arquivo GZ: {e}")
            return None
            
    # 4. Arquivo CSV normal (simplesmente salva para que o agente possa ler do disco)
    elif file_extension == 'csv':
        try:
            with open(tmp_csv_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            return tmp_csv_path
        except Exception as e:
            st.sidebar.error(f"Erro ao salvar o arquivo CSV: {e}")
            return None
    
    else:
        st.sidebar.error("Formato de arquivo não suportado. Use CSV, ZIP ou GZ.")
        return None

def create_data_agent(_llm, _csv_path):
    """Cria o agente CSV usando o caminho do arquivo."""
    try:
        return create_csv_agent(
            _llm, 
            _csv_path, 
            verbose=True, 
            agent_type="zero-shot-react-description", # Tipo de agente mais robusto para Gemini
            allow_dangerous_code=True,
            handle_parsing_errors=True 
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o agente de IA: {e}")
        return None

# --- Interface e Execução ---

st.title("🛡️ Multi Agente Analista de CSV Fiscal")
st.markdown("Faça o upload de qualquer arquivo CSV, ZIP ou GZ para começar a análise fiscal e contábil. Defina regras personalizadas na barra lateral.")


# --- BARRA LATERAL (Sidebar) ---
st.sidebar.header("Configuração do Analista")

# 1. Componente de Upload
uploaded_file = st.sidebar.file_uploader(
    "1. Faça o Upload do Arquivo (CSV, ZIP ou GZ)", 
    type=["csv", "zip", "gz"]
)

# 2. Componente de Instruções Personalizadas
custom_instructions = st.sidebar.text_area(
    "2. Instruções/Regras Específicas para o Agente",
    "Ex: Focar apenas em transações acima de R$10.000 ou analisar o desempenho do 'Departamento Contas a Pagar'.",
    height=150
)

# Inicializa o LLM e a Memória com as instruções personalizadas
llm, conversation_chain, memory = load_llm_and_memory(custom_instructions)

data_agent_executor = None

if uploaded_file is not None and llm:
    # Processa e descompacta o arquivo se necessário
    tmp_file_path = decompress_file(uploaded_file)

    if tmp_file_path:
        # 3. Cria o agente com o caminho temporário
        data_agent_executor = create_data_agent(llm, tmp_file_path)

        # Armazena o agente na sessão
        if 'data_agent' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.data_agent = data_agent_executor
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = [] # Limpa o chat para o novo arquivo
            st.sidebar.success(f"Arquivo '{uploaded_file.name}' carregado e agente inicializado!")

            # Simula a pergunta inicial (O Agente precisa começar com uma tarefa para evitar o erro de parsing)
            initial_q1 = (
                f"O arquivo {uploaded_file.name} foi carregado. Suas instruções são: '{custom_instructions}'. "
                f"Agora, descreva os dados: Quais são as colunas, tipos de dados e número de linhas/colunas?"
            )
            st.session_state.messages.append({"role": "user", "content": initial_q1})
            
            with st.spinner(f"Agente pensando em: {initial_q1[:50]}..."):
                try:
                    # O agente executa a análise inicial
                    # Usamos o método .run() que é mais adequado para a primeira pergunta do zero-shot-react-description
                    response = st.session_state.data_agent.run(initial_q1)
                except Exception as e:
                    # Tratamento de erro robusto para a inicialização
                    response = (
                        f"O agente inicializou, mas encontrou um erro ao tentar a primeira análise. "
                        f"Tente perguntar novamente ou faça uma pergunta mais simples. Detalhes: {e}"
                    )
                    st.error(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Mostra o cabeçalho do arquivo
            try:
                df_preview = pd.read_csv(tmp_file_path, low_memory=False)
                st.subheader(f"Pré-visualização do Arquivo: {uploaded_file.name}")
                st.dataframe(df_preview.head())
            except Exception as e:
                 st.warning(f"Não foi possível exibir a pré-visualização do arquivo. Detalhes: {e}")
            
            # Remove o arquivo temporário
            os.remove(tmp_file_path)
            
# --- Bloco de Geração de Relatório e Chat ---

if 'data_agent' in st.session_state and st.session_state.data_agent:
    st.markdown("---")
    
    # 3. Botão de Geração de Relatório na Barra Lateral
    if st.sidebar.button("📝 Gerar Relatório Completo", type="primary"):
        # Prompt usa a memória da ConversationChain para resumir a sessão
        report_prompt = (
            "Gere um relatório completo e estruturado em Markdown, com base em todas as análises realizadas até agora "
            "nesta sessão de chat. Inclua uma Introdução (mencionando o arquivo analisado e as instruções do usuário), "
            "Análise Estatística (com base nas perguntas feitas e respostas), e uma Conclusão final sobre os dados. "
            "NÃO use a ferramenta CSV Agent para isso, apenas resuma o histórico do chat."
        )
        with st.spinner("Gerando relatório final e estruturado..."):
            try:
                # O ConversationChain usa o histórico (chat_history)
                final_report = conversation_chain.run(report_prompt)
                st.session_state['final_report_content'] = final_report
                st.sidebar.success("Relatório gerado com sucesso! Baixe abaixo.")
            except Exception as e:
                st.sidebar.error(f"Erro ao gerar o relatório: {e}")

    # 4. Download button para o relatório gerado na Barra Lateral
    if 'final_report_content' in st.session_state:
        st.sidebar.download_button(
            label="⬇️ Baixar Relatório em Markdown",
            data=st.session_state['final_report_content'],
            file_name=f"Relatorio_{st.session_state.current_file.replace('.csv', '').replace('.', '')}.md",
            mime="text/markdown"
        )
        st.markdown("---")


    # --- Exibição do Histórico do Chat ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            
            # Tenta analisar se o conteúdo é um JSON de gráfico Plotly
            is_plotly_json = False
            try:
                # Remove qualquer ruído (como 'Final Answer: ' que o agente pode adicionar)
                clean_content = content.strip().replace("Final Answer:", "").strip()
                
                # Tenta carregar o conteúdo como JSON
                fig_json = json.loads(clean_content)
                
                # Verifica se é um objeto Plotly válido
                if 'data' in fig_json and 'layout' in fig_json:
                    fig = pio.from_json(clean_content)
                    st.plotly_chart(fig, use_container_width=True)
                    is_plotly_json = True
                
            except (json.JSONDecodeError, ValueError, KeyError):
                # Não é um JSON de gráfico Plotly, então continua
                pass

            if not is_plotly_json:
                st.markdown(content)
    
    # --- Entrada do Usuário (Bloco de Chat) ---
    if prompt := st.chat_input("Faça sua pergunta ao Agente Analista (Ex: Calcule a média da coluna 'Valor' e gere um histograma dela):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agente de IA está analisando os dados..."):
                try:
                    # Envia a pergunta para o agente de dados
                    # O método .run() é usado aqui para o agente de dados
                    response = st.session_state.data_agent.run(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    # Este bloco de exceção captura e exibe erros, permitindo que o app continue
                    error_message = (
                        f"O Agente encontrou um erro ao processar sua requisição. Por favor, tente reformular a pergunta "
                        f"ou verificar se o arquivo CSV está bem formatado. Detalhes: {e}"
                    )
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Por favor, faça o upload de um arquivo CSV, ZIP ou GZ para começar a análise.")
