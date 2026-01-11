"""
Basic tests for the RAG system
"""

def test_document_loader():
    """Test document loading"""
    from src.document_processor import DocumentLoader
    loader = DocumentLoader()
    assert loader.supported_formats == ['.pdf', '.txt', '.docx']
    print("✅ DocumentLoader test passed")

def test_text_splitter():
    """Test text splitting"""
    from src.document_processor import TextSplitter
    splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
    
    test_doc = {
        "source": "test.txt",
        "content": "This is a test document. " * 50
    }
    
    chunks = splitter.split_text(test_doc)
    assert len(chunks) > 0
    assert all('content' in chunk for chunk in chunks)
    print(f"✅ TextSplitter test passed - Created {len(chunks)} chunks")

if __name__ == "__main__":
    print("Running tests...\n")
    test_document_loader()
    test_text_splitter()
    print("\n🎉 All tests passed!")
