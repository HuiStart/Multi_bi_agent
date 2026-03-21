import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_ollama import OllamaEmbeddings

load_dotenv()

class Config:
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Configs
    # MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "tongyi")
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")
    
    # MODEL_NAME = os.getenv("MODEL_NAME", "qwen-max")
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5")

    @staticmethod
    def get_llm(temperature=0.3):
        provider = Config.MODEL_PROVIDER.lower()
        if provider == "openai":
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                model=Config.MODEL_NAME,
                temperature=temperature
            )
        elif provider == "deepseek":
            return ChatOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                temperature=temperature
            )
        elif provider == "tongyi":
            # Aliyun DashScope offers OpenAI-compatible endpoint
            return ChatOpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=os.getenv('DASHSCOPE_MODEL_NAME'),
                temperature=temperature
            )
        elif provider == "ollama":
            # Ollama offers OpenAI-compatible endpoint at /v1
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            
            return ChatOpenAI(
                api_key="ollama", # Placeholder
                base_url=base_url,
                model=os.getenv("OLLAMA_MODEL_NAME", "qwen2.5"),
                temperature=temperature
            )
        elif provider == "wenxin":
            return QianfanChatEndpoint(
                qianfan_ak=os.getenv("QIANFAN_AK"),
                qianfan_sk=os.getenv("QIANFAN_SK"),
                model=os.getenv("MODEL_NAME", "ERNIE-Bot"),
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    @staticmethod
    def get_embeddings():
        provider = Config.MODEL_PROVIDER.lower()
        if provider == "openai":
            return OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            )
        elif provider == "tongyi":
            # Aliyun DashScope embeddings are also OpenAI-compatible
            return OpenAIEmbeddings(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=os.getenv('DASHSCOPE_EMBED_MODEL', 'text-embedding-v3')
            )
        elif provider == "ollama":
            return OllamaEmbeddings(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").replace("/v1", ""),
                model=os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:22m")
            )
        else:
            # Fallback to OpenAI if key exists, otherwise local
            if os.getenv("OPENAI_API_KEY"):
                return OpenAIEmbeddings()
            else:
                raise ValueError(f"No embedding configuration for provider: {provider}")

# For testing, you might want to mock the LLM or provide a simple one
