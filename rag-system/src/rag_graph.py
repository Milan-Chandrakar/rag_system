from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.vector_store import VectorStore
from config.settings import settings

class RAGState(TypedDict):
    """State for the RAG graph"""
    messages: Annotated[List[HumanMessage | AIMessage], "Conversation messages"]
    query: str
    retrieved_docs: List[dict]
    context: str
    answer: str
    sources: List[str]

class RAGGraph:
    """LangGraph-based RAG system"""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize RAG graph with LLM and vector store"""
        self.llm = ChatGroq(
            temperature=settings.llm_temperature,
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model
        )
        
        self.vector_store = vector_store
        
        # Define the RAG prompt
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer the user's question based on the context provided below.
            
Context:
{context}

If the context doesn't contain relevant information to answer the question, say so honestly.
Provide clear, concise answers and cite the sources when relevant."""),
            ("human", "{query}"),
        ])
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the RAG workflow graph"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vector store"""
        query = state['query']
        
        # Search vector store
        docs = self.vector_store.search(query, top_k=settings.top_k_results)
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc['content']}")
            sources.append(doc['metadata']['source'])
        
        context = "\n\n".join(context_parts)
        
        return {
            **state,
            "retrieved_docs": docs,
            "context": context,
            "sources": list(set(sources)),
        }
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using LLM"""
        # Format the prompt
        messages = self.rag_prompt.format_messages(
            context=state['context'],
            query=state['query']
        )
        
        # Get LLM response
        response = self.llm.invoke(messages)
        
        # Update state
        state['answer'] = response.content
        state['messages'].append(HumanMessage(content=state['query']))
        state['messages'].append(AIMessage(content=response.content))
        
        return state
    
    def query(self, question: str) -> dict:
        """Process a query through the RAG pipeline"""
        # Initialize state
        initial_state = {
            "messages": [],
            "query": question,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result['answer'],
            "sources": result['sources'],
            "num_docs_retrieved": len(result['retrieved_docs'])
        }
