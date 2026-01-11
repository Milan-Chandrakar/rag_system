from pathlib import Path
from typing import List, Dict
import pypdf
from docx import Document as DocxDocument
import tiktoken

class DocumentLoader:
    """Load documents from various file formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def load_document(self, file_path: Path) -> Dict[str, str]:
        """Load a document and return its content"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._load_pdf(file_path)
        elif suffix == '.txt':
            return self._load_txt(file_path)
        elif suffix == '.docx':
            return self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_pdf(self, file_path: Path) -> Dict[str, str]:
        """Load PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return {"source": str(file_path.name), "content": text}
    
    def _load_txt(self, file_path: Path) -> Dict[str, str]:
        """Load TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return {"source": str(file_path.name), "content": text}
    
    def _load_docx(self, file_path: Path) -> Dict[str, str]:
        """Load DOCX file"""
        doc = DocxDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return {"source": str(file_path.name), "content": text}


class TextSplitter:
    """Split text into chunks for embedding"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def split_text(self, document: Dict[str, str]) -> List[Dict[str, str]]:
        """Split document into chunks"""
        text = document["content"]
        source = document["source"]
        
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                "content": chunk_text,
                "source": source,
                "chunk_id": len(chunks)
            })
        
        return chunks
