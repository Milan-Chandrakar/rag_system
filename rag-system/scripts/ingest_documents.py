import sys
from pathlib import Path
from src.document_processor import DocumentLoader, TextSplitter
from src.vector_store import VectorStore
from config.settings import settings

def ingest_document(file_path: str):
    """Ingest a single document into the vector store"""
    loader = DocumentLoader()
    splitter = TextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    vector_store = VectorStore()
    
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📄 Loading document: {path.name}")
    document = loader.load_document(path)
    
    print(f"✂️ Splitting into chunks...")
    chunks = splitter.split_text(document)
    
    print(f"💾 Adding {len(chunks)} chunks to vector store...")
    vector_store.add_documents(chunks)
    
    print(f"\n✅ Successfully ingested: {path.name}")
    print(f"📊 Total documents in database: {vector_store.get_collection_count()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_documents.py <file_path>")
        sys.exit(1)
    
    ingest_document(sys.argv[1])
