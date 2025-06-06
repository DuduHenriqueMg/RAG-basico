import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, docs_path="documentos"):
        """Carrega todos os PDFs da pasta documentos"""
        documents = []
        pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))
        
        if not pdf_files:
            st.error(f"Nenhum PDF encontrado na pasta {docs_path}")
            return documents
            
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Carregado: {os.path.basename(pdf_file)}")
            except Exception as e:
                st.error(f"❌ Erro ao carregar {pdf_file}: {str(e)}")
                
        return documents
    
    def split_documents(self, documents):
        """Divide os documentos em chunks menores"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_documents(documents)
    
    def create_vectorstore(self, chunks):
        """Cria o banco vetorial com os chunks"""
        try:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            st.success(f"✅ Base vetorial criada com {len(chunks)} chunks")
            return True
        except Exception as e:
            st.error(f"❌ Erro ao criar base vetorial: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """Configura a cadeia de pergunta e resposta"""
        if not self.vectorstore:
            return False
            
        # Template personalizado para respostas em português
        template = """
        Use as seguintes informações para responder à pergunta do usuário.
        Se você não souber a resposta baseada no contexto fornecido, diga que não sabe.
        
        Contexto: {context}
        
        Pergunta: {question}
        
        Resposta detalhada:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return True
    
    def query(self, question):
        """Faz uma pergunta ao sistema RAG"""
        if not self.qa_chain:
            return "Sistema não inicializado. Carregue os documentos primeiro.", []
            
        try:
            result = self.qa_chain({"query": question})
            return result["result"], result.get("source_documents", [])
        except Exception as e:
            return f"Erro ao processar pergunta: {str(e)}", []
    
    def initialize_system(self):
        """Inicializa todo o sistema RAG"""
        with st.spinner("Carregando documentos..."):
            documents = self.load_documents()
            
        if not documents:
            return False
            
        with st.spinner("Processando documentos..."):
            chunks = self.split_documents(documents)
            
        with st.spinner("Criando base vetorial..."):
            if not self.create_vectorstore(chunks):
                return False
                
        with st.spinner("Configurando sistema de perguntas..."):
            if not self.setup_qa_chain():
                return False
                
        st.success("🎉 Sistema RAG inicializado com sucesso!")
        return True