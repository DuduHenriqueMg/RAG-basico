import streamlit as st
import os
from rag_system import RAGSystem

# Configuração da página
st.set_page_config(
    page_title="Chatbot RAG - Documentos PDF",
    page_icon="🤖",
    layout="wide"
)

# Título principal
st.title("🤖 Chatbot RAG para Documentos PDF")
st.markdown("---")

# Inicialização do sistema RAG
@st.cache_resource
def initialize_rag():
    return RAGSystem()

# Função para verificar se a API key está configurada
def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Configure sua OPENAI_API_KEY no arquivo .env")
        st.stop()

# Verificar API key
check_api_key()

# Inicializar sistema RAG
rag_system = initialize_rag()

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Verificar se a pasta documentos existe
    docs_folder = "documentos"
    if not os.path.exists(docs_folder):
        st.error(f"📁 Pasta '{docs_folder}' não encontrada!")
        st.info("Crie a pasta 'documentos' e adicione seus PDFs")
        st.stop()
    
    # Listar PDFs na pasta
    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]
    
    if pdf_files:
        st.success(f"📄 {len(pdf_files)} PDF(s) encontrado(s):")
        for pdf in pdf_files:
            st.write(f"• {pdf}")
    else:
        st.warning("Nenhum PDF encontrado na pasta documentos")
        st.stop()
    
    st.markdown("---")
    
    # Botão para inicializar o sistema
    if st.button("🚀 Inicializar Sistema RAG", type="primary"):
        if rag_system.initialize_system():
            st.session_state.rag_initialized = True
        else:
            st.session_state.rag_initialized = False
    
    # Status do sistema
    if st.session_state.get('rag_initialized', False):
        st.success("✅ Sistema RAG ativo")
    else:
        st.warning("⏳ Sistema não inicializado")

# Interface principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat")
    
    # Verificar se o sistema está inicializado
    if not st.session_state.get('rag_initialized', False):
        st.info("👈 Clique em 'Inicializar Sistema RAG' na barra lateral para começar")
    else:
        # Inicializar histórico de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Exibir histórico de mensagens
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input do usuário
        if prompt := st.chat_input("Faça sua pergunta sobre os documentos..."):
            # Adicionar mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Gerar resposta
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    response, sources = rag_system.query(prompt)
                
                st.markdown(response)
                
                # Mostrar fontes se disponíveis
                if sources:
                    with st.expander("📚 Fontes consultadas"):
                        for i, source in enumerate(sources):
                            st.write(f"**Fonte {i+1}:**")
                            st.write(f"Arquivo: {source.metadata.get('source', 'N/A')}")
                            st.write(f"Página: {source.metadata.get('page', 'N/A')}")
                            st.write(f"Conteúdo: {source.page_content[:200]}...")
                            st.markdown("---")
            
            # Adicionar resposta ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.header("ℹ️ Informações")
    
    st.markdown("""
    ### Como usar:
    1. **Adicione PDFs** na pasta `documentos/`
    2. **Clique** em "Inicializar Sistema RAG"
    3. **Faça perguntas** sobre o conteúdo dos documentos
    
    ### Exemplos de perguntas:
    - "Qual o resumo do documento?"
    - "Quais são os pontos principais?"
    - "Explique sobre [tópico específico]"
    
    ### Recursos:
    - ✅ Busca semântica nos documentos
    - ✅ Respostas contextualizadas
    - ✅ Fontes das informações
    - ✅ Histórico de conversas
    """)
    
    # Botão para limpar histórico
    if st.session_state.get('rag_initialized', False):
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.messages = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit + LangChain + OpenAI")