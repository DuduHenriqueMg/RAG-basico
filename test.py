import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def test_openai_connection():
    """Testa a conexão com a API da OpenAI"""
    
    # Verificar se a chave existe
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY não encontrada no arquivo .env")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ OPENAI_API_KEY parece estar incorreta (deve começar com 'sk-')")
        return False
    
    print(f"✅ API Key encontrada: {api_key[:10]}...")
    
    # Testar embeddings
    try:
        print("🔄 Testando conexão com OpenAI...")
        embeddings = OpenAIEmbeddings()
        
        # Teste simples
        test_text = ["Teste de conexão"]
        result = embeddings.embed_documents(test_text)
        
        print("✅ Conexão com OpenAI funcionando!")
        print(f"✅ Embedding gerado com {len(result[0])} dimensões")
        return True
        
    except Exception as e:
        print(f"❌ Erro na conexão: {str(e)}")
        
        # Diagnósticos específicos
        if "authentication" in str(e).lower():
            print("💡 Problema: API Key inválida ou expirada")
        elif "quota" in str(e).lower():
            print("💡 Problema: Cota da API esgotada")
        elif "connection" in str(e).lower():
            print("💡 Problema: Erro de rede/proxy")
        
        return False

if __name__ == "__main__":
    print("🧪 Testando conexão com OpenAI...")
    print("=" * 50)
    test_openai_connection()