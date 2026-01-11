import gradio as gr
from pathlib import Path
from src.rag_graph import RAGGraph
from src.document_processor import DocumentLoader, TextSplitter
from src.vector_store import VectorStore
from config.settings import settings

# Initialize components
print("🚀 Initializing RAG System...")
vector_store = VectorStore()
rag_graph = RAGGraph(vector_store)
document_loader = DocumentLoader()
text_splitter = TextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
print("✅ RAG System Ready!")


def query_rag(question: str) -> tuple[str, str]:
    """Query the RAG system"""
    if not question.strip():
        return "Please enter a question.", ""
    
    # Check if there are documents in the vector store
    doc_count = vector_store.get_collection_count()
    if doc_count == 0:
        return ("⚠️ No documents in the knowledge base. Please upload documents first "
                "using the 'Document Management' tab."), ""
    
    # Query the RAG system
    result = rag_graph.query(question)
    
    # Format the response
    answer = result['answer']
    sources_text = "\n".join([f"📄 {source}" for source in result['sources']])
    
    if not sources_text:
        sources_text = "No sources found"
    
    return answer, sources_text


def upload_documents(files) -> str:
    """Upload and process documents"""
    if not files:
        return "⚠️ No files uploaded"
    
    try:
        total_chunks = 0
        processed_files = []
        
        for file in files:
            file_path = Path(file.name)
            
            # Load document
            document = document_loader.load_document(file_path)
            
            # Split into chunks
            chunks = text_splitter.split_text(document)
            
            # Add to vector store
            vector_store.add_documents(chunks)
            
            total_chunks += len(chunks)
            processed_files.append(file_path.name)
        
        doc_count = vector_store.get_collection_count()
        
        status = f"""✅ Successfully processed {len(files)} document(s)

📊 Details:
- Files processed: {', '.join(processed_files)}
- Chunks created: {total_chunks}
- Total documents in database: {doc_count}
"""
        return status
        
    except Exception as e:
        return f"❌ Error processing documents: {str(e)}"


def get_database_stats() -> str:
    """Get current database statistics"""
    doc_count = vector_store.get_collection_count()
    return f"""📊 Knowledge Base Statistics:

- Total document chunks: {doc_count}
- Vector store: ChromaDB
- Embedding model: {settings.embedding_model}
- LLM model: {settings.llm_model}
- Retrieval: Top {settings.top_k_results} documents
"""


# Build the Gradio interface
with gr.Blocks(theme='Yntec/HaleyCH_Theme_Orange_Green', title="RAG System") as interface:
    gr.Markdown("""
    # 🤖 Production RAG System
    ### Retrieval-Augmented Generation with LangGraph
    
    Upload documents (PDF, TXT, DOCX) and ask questions about them!
    """)
    
    with gr.Tabs():
        # Query Tab
        with gr.Tab("💬 Ask Questions"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Enter your question",
                        placeholder="What would you like to know about your documents?",
                        lines=3
                    )
                    query_btn = gr.Button("🔍 Ask", variant="primary", size="lg")
                
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=10,
                        show_copy_button=True
                    )
                with gr.Column():
                    sources_output = gr.Textbox(
                        label="Sources",
                        lines=10
                    )
            
            gr.Examples(
                examples=[
                    ["What are the main topics covered in the documents?"],
                    ["Summarize the key findings"],
                    ["What recommendations are mentioned?"],
                ],
                inputs=question_input
            )
        
        # Document Management Tab
        with gr.Tab("📁 Document Management"):
            gr.Markdown("### Upload Documents")
            gr.Markdown("Supported formats: PDF, TXT, DOCX")
            
            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".docx"]
            )
            upload_btn = gr.Button("📤 Upload & Process", variant="primary")
            upload_status = gr.Textbox(label="Status", lines=8)
            
            gr.Markdown("### Database Statistics")
            stats_btn = gr.Button("📊 Refresh Statistics")
            stats_output = gr.Textbox(label="Current Statistics", lines=6)
    
    # Event handlers
    query_btn.click(
        fn=query_rag,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )
    
    question_input.submit(
        fn=query_rag,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )
    
    upload_btn.click(
        fn=upload_documents,
        inputs=[file_upload],
        outputs=[upload_status]
    )
    
    stats_btn.click(
        fn=get_database_stats,
        outputs=[stats_output]
    )
    
    # Load initial stats
    interface.load(fn=get_database_stats, outputs=[stats_output])


def launch_app():
    """Launch the Gradio application"""
    interface.launch(
        share=settings.gradio_share,
        server_port=settings.gradio_server_port
    )


if __name__ == "__main__":
    launch_app()
