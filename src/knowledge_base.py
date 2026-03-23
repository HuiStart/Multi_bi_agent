import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .config import Config

class KnowledgeBaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or Config.CHROMA_DB_PATH
        self.embeddings = Config.get_embeddings()
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def add_document(self, file_path):
        """Upload and process a document."""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        docs = loader.load()
        splits = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(splits)
        return len(splits)

    def retrieve_knowledge(self, query, k=3):
        """Retrieve relevant knowledge from the vector store."""
        results = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])

    def update_knowledge(self, content, source="Manual Update"):
        """Manually update the knowledge base with new text content."""
        from langchain_core.documents import Document
        doc = Document(page_content=content, metadata={"source": source})
        splits = self.text_splitter.split_documents([doc])
        self.vector_store.add_documents(splits)
        return len(splits)

    def update_knowledge_batch(self, contents, batch_size=100):
        """Batch update knowledge base for better performance."""
        from langchain_core.documents import Document
        docs = [Document(page_content=content, metadata={"source": source})
                for content, source in contents]
        splits = self.text_splitter.split_documents(docs)

        # Add in batches
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            self.vector_store.add_documents(batch)
            print(f"  Added {min(i + batch_size, len(splits))}/{len(splits)} chunks")

        return len(splits)
