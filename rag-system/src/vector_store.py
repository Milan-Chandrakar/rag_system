from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from config.settings import settings

class VectorStore:
    """Manage vector embeddings and similarity search"""
    
    def __init__(self):
        """Initialize ChromaDB and embedding model"""
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name
        )
        self.embedding_model = SentenceTransformer(settings.embedding_model)
    
    def add_documents(self, chunks: List[Dict[str, str]]):
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        ids = [f"{chunk['source']}_chunk_{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [{"source": chunk['source'], "chunk_id": chunk['chunk_id']} 
                     for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents"""
        top_k = top_k or settings.top_k_results
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                documents.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return documents
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        return self.collection.count()
