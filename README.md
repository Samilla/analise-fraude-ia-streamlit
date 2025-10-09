# analise-fraude-ia-streamlit
🤖 Multi Agente de Análise Fiscal e de Fraudes
Sistema inteligente de análise de dados fiscais com IA, otimizado para detecção de fraudes, análise exploratória automática e geração de insights através de linguagem natural.
✨ Características Principais

🧠 Agente IA com Gemini 2.0 Flash: Análise conversacional de dados usando linguagem natural
📊 Geração Automática de Gráficos: Detecta automaticamente quando criar visualizações (barras, linha, dispersão, pizza, histograma, boxplot)
💾 Sistema de Cache Inteligente: Economiza tokens reutilizando respostas (TTL de 1 hora)
🔒 Rate Limiting: Proteção contra uso excessivo (100 chamadas/dia, 2s entre requisições)
📈 Análise Exploratória Completa: Relatórios automáticos com insights estatísticos
🎯 Detecção de Anomalias: Identificação de padrões suspeitos e outliers
📁 Suporte Multi-Formato: CSV, ZIP (com CSV interno) e arquivos comprimidos GZ

🚀 Instalação
Pré-requisitos

Python 3.12 ou superior
Chave de API do Google Gemini (obtenha aqui)

Passo a Passo

1. Clone o repositório

bashgit clone https://github.com/seu-usuario/multi-agente-fiscal.git
cd multi-agente-fiscal

2.Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

3. Instale as dependências
pip install -r requirements.txt

4.Configure a API Key

Crie um arquivo .streamlit/secrets.toml:
GEMINI_API_KEY = "sua-chave-api-aqui"

Ou configure como variável de ambiente:
export GEMINI_API_KEY="sua-chave-api-aqui"  # Linux/Mac
set GEMINI_API_KEY=sua-chave-api-aqui  # Windows

5.Execute a aplicação
streamlit run app.py

📦 Dependências
streamlit
pandas
plotly
langchain
langchain-community
google-generativeai
langchain-google-genai
langchain-experimental
tabulate
seaborn

Crie um arquivo requirements.txt com o conteúdo acima.

💡 Como Usar
1. Carregamento de Dados

Clique em "📁 Carregue seu arquivo" na barra lateral
Suporta arquivos CSV, ZIP (contendo CSV) ou GZ
O sistema carrega e processa automaticamente

2. Análise Conversacional
Faça perguntas em linguagem natural:
"Quantas linhas tem o dataset?"
"Qual o valor total da coluna x?
"Mostre a distribuição dos valores"
"Gráfico de barras da média da coluna x e y"
"Identifique valores atípicos na coluna x"

 3. Geração Automática de Gráficos
O sistema detecta automaticamente quando criar visualizações:

Gráfico de Barras: "mostre em barras", "média da coluna x e y"
Gráfico de Linha: "evolução temporal", "tendência ao longo do tempo"
Dispersão: "correlação entre X e Y", "relação entre variáveis"
Pizza: "proporção", "distribuição percentual"
Histograma: "distribuição de valores", "frequência"
Boxplot: "outliers", "valores atípicos"

4. Relatório Automático
Clique em "🔄 Gerar Relatório" na sidebar para obter:

Total de linhas e colunas
Estatísticas descritivas
Categorias mais frequentes
Detecção de valores nulos
Insights e anomalias

🎯 Exemplos de Uso
Análise Exploratória Básica
# Perguntas iniciais
"Quais são as colunas do dataset?"
"Mostre as primeiras 10 linhas"
"Existem valores nulos?"
"Qual a estatística descritiva das colunas numéricas?"

Análise Fiscal
python# Análise de notas fiscais
•	Quais são as colunas do dataset?
•	Quantas linhas e colunas existem?
•	Quais são os tipos de dados numéricos e categóricos?
•	Quais as medidas de tendência central (média)?
•	Quais as medidas de tendência central (moda)?
•	Qual a variabilidade dos dados (desvio padrão)?

Detecção de Fraudes
python# Identificação de anomalias
"Identifique valores atípicos na coluna x"
"Existem transações duplicadas?"
"Boxplot dos valores para detectar outliers"
Qual a variabilidade dos dados da coluna x e y ?
Faça um gráfico de dispersão entre x e y


⚙️ Configurações Avançadas
Limites e Otimizações

O sistema possui configurações ajustáveis no código:
pythonMAX_DAILY_CALLS = 100        # Limite de chamadas diárias à API
MIN_CALL_INTERVAL = 2        # Segundos entre chamadas
CACHE_TTL = 3600            # Tempo de vida do cache (segundos)
MAX_TOKENS = 1024           # Tokens máximos por resposta
MAX_HISTORY_SIZE = 10       # Mensagens mantidas em memória

Economia de Tokens

Cache automático: Respostas idênticas são reutilizadas
Rate limiting: Previne uso excessivo
Histórico limitado: Apenas últimas 10 mensagens em contexto
Respostas concisas: Máximo 3-4 linhas por resposta

📊 Métricas e Monitoramento
A sidebar exibe em tempo real:

🔥 Chamadas Hoje: Quantas requisições foram feitas
💾 Cache Hits: Quantas respostas vieram do cache
💰 Economia: Percentual de economia de tokens
📊 Progresso: Barra de uso do limite diário

🔧 Solução de Problemas
Erro: "Chave da API não encontrada"

Verifique se configurou GEMINI_API_KEY corretamente
Confirme que o arquivo .streamlit/secrets.toml existe

Erro: "Limite diário atingido"

O contador reseta automaticamente à meia-noite
Use o cache! Refaça perguntas anteriores
Clique em "🔄 Reset Total" para limpar tudo (cuidado!)

Erro de parsing

Reformule a pergunta de forma mais simples
Exemplo: "Mostre as 5 primeiras linhas"
Evite perguntas muito complexas ou longas

Gráficos não aparecem

Especifique as colunas: "gráfico de barras da coluna Status"
Use palavras-chave: "mostre em barras", "plote", "visualize"

🤝 Contribuindo
Contribuições são bem-vindas! Para contribuir:

Faça um Fork do projeto
Crie uma branch para sua feature (git checkout -b feature/MinhaFeature)
Commit suas mudanças (git commit -m 'Adiciona MinhaFeature')
Push para a branch (git push origin feature/MinhaFeature)
Abra um Pull Request

📝 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

👨‍💻 Autor: SAMILLA MACEDO
Desenvolvido usando Streamlit, LangChain e Google Gemini

🔗 Links Úteis

Documentação do Streamlit
LangChain Documentation
Google Gemini API
Plotly Documentation

🎓 Casos de Uso

Auditoria Fiscal: Análise de notas fiscais e detecção de irregularidades
Análise de Fraudes: Identificação de padrões suspeitos em transações
Business Intelligence: Geração de insights a partir de dados comerciais
Análise Exploratória: Investigação inicial de qualquer dataset
Relatórios Automáticos: Documentação instantânea de análises
