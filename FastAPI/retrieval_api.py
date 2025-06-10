# VectorDB
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VectorDB.chroma_db import ChromaDB

# Langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# FastAPI
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Formatting
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

# Other
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Knowledge Base API",
    description="A FastAPI backend for RAG-based question answering with source references",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    """Model for storing the query request."""

    question: str = Field(..., min_length=1, max_length=1000, description="User question to be answered")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()

class QueryResponse(BaseModel):
    """Model for storing the query response."""

    answer: str = Field(..., description="Generated answer from RAG system")
    sources: List[str] = Field(..., description="Distinct source URLs used to generate the answer")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the response")

class StatsResponse(BaseModel):
    """Model for storing statistics about the knowledge base."""

    total_documents: int = Field(..., description="Total number of documents in the knowledge base")
    total_chunks: int = Field(..., description="Total number of text chunks in the knowledge base")
    collection_name: str = Field(..., description="Name of the ChromaDB collection")
    programming_languages: Dict[str, int] = Field(..., description="Count of chunks per programming language")
    sections: Dict[str, int] = Field(..., description="Count of chunks per section")
    levels: Dict[str, int] = Field(..., description="Count of chunks per level")
    unique_urls: List[str] = Field(..., description="List of unique URLs in the knowledge base")
    avg_chunk_length: float = Field(..., description="Average length of text chunks")
    total_characters: int = Field(..., description="Total number of characters in all chunks")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")

class AssistantResponse(BaseModel):
    """Model for storing query analysis results with programming language detection."""
    
    questions: List[str] = Field(..., description="Three different questions generated from user input")
    has_code: bool = Field(default=False, description="Whether user mentions code")
    has_curl: bool = Field(default=False, description="Whether user mentions curl")
    has_python: bool = Field(default=False, description="Whether user mentions python")
    has_php: bool = Field(default=False, description="Whether user mentions php")
    has_csharp: bool = Field(default=False, description="Whether user mentions csharp")
    has_golang: bool = Field(default=False, description="Whether user mentions golang")
    has_java: bool = Field(default=False, description="Whether user mentions java")
    has_javascript: bool = Field(default=False, description="Whether user mentions javascript")
    has_json: bool = Field(default=False, description="Whether user mentions json")
    has_html: bool = Field(default=False, description="Whether user mentions html")
    has_sql: bool = Field(default=False, description="Whether user mentions sql")

class ErrorResponse(BaseModel):
    """Model for storing error response."""

    error: str = Field(..., description="Error message")
    detail: str = Field(..., description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class InformationRetrieval: 
    def __init__(self, user_query: str) -> None:
        """Initialize retrieval system with user query and configure LLM connection."""

        self.chroma_db = ChromaDB(persist_directory="/Users/edvardsivickij/Documents/My documents/oxylabs/VectorDB/chroma_storage")
        self.llm = ChatOpenAI(
            model="gpt-4.1", 
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        self.user_query = user_query
  
    def initialize_chroma_db(self) -> Chroma:
        """Initialize ChromaDB client with embedding model for vector search."""
        
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        return Chroma(
            client=self.chroma_db.client,
            collection_name=self.chroma_db.collection_name,
            embedding_function=embedding_model
        )
    
    async def initial_bot(self, user_query: str) -> AssistantResponse:
        """Transform user query into optimized search questions with language detection."""
        
        system_prompt = """
        ### Role
        You are an intelligent query optimizer. Your task is to transform a single user query into three semantically diverse questions that maximize retrieval accuracy from a vector database.
        
        ### Problem
        Vector databases rely on semantic similarity, which can miss relevant results if the query phrasing doesn't match stored content. Multiple query variations increase the likelihood of finding the correct information.
        
        ###Task
        Transform the user's input into three distinct questions that capture different aspects or phrasings of the same intent.
        Detect if the query references code/programming and set has_code: true.
        Identify specific programming languages from: [curl, python, php, csharp, golang, java, javascript, json, html, sql] and set corresponding flags.

        ### Output Format
        Return a JSON object with this exact structure:
        {{
        questions": ["question1", "question2", "question3"],
        has_code": true/false,
        has_curl": true/false,
        has_python": true/false,
        has_php": true/false,
        has_csharp": true/false,
        has_golang": true/false,
        has_java": true/false,
        has_javascript": true/false,
        has_json": true/false,
        has_html": true/false,
        has_sql": true/false
        }}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original query: {user_query}")
        ]
        
        response = await self.llm.ainvoke(messages)
        response_dict = json.loads(response.content)
        validated_response = AssistantResponse(**response_dict)
        return validated_response
    
    async def information_retrieval(self, assistant_response: AssistantResponse, chroma: Chroma) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using vector search and BM25 ensemble approach."""
        
        all_vector_results = []
        
        for question in assistant_response.questions:
            if assistant_response.has_code:
                code_filters = []
                if assistant_response.has_curl:
                    code_filters.append({"has_curl": True})
                if assistant_response.has_python:
                    code_filters.append({"has_python": True})
                if assistant_response.has_php:
                    code_filters.append({"has_php": True})
                if assistant_response.has_csharp:
                    code_filters.append({"has_csharp": True})
                if assistant_response.has_golang:
                    code_filters.append({"has_golang": True})
                if assistant_response.has_java:
                    code_filters.append({"has_java": True})
                if assistant_response.has_javascript:
                    code_filters.append({"has_javascript": True})
                if assistant_response.has_json:
                    code_filters.append({"has_json": True})
                if assistant_response.has_html:
                    code_filters.append({"has_html": True})
                if assistant_response.has_sql:
                    code_filters.append({"has_sql": True})
                
                specific_results = []
                
                if code_filters:
                    specific_filter = {"$or": code_filters} if len(code_filters) > 1 else code_filters[0]
                    specific_retriever = chroma.as_retriever(
                        search_type="similarity",
                        search_kwargs={'k': 10, 'filter': specific_filter}
                    )
                    specific_results = await specific_retriever.ainvoke(question)
                    all_vector_results.extend(specific_results)
                
                # Search with general code filter only if no specific results found
                if len(specific_results) == 0:
                    general_retriever = chroma.as_retriever(
                        search_type="similarity",
                        search_kwargs={'k': 10, 'filter': {"has_code": True}}
                    )
                    general_results = await general_retriever.ainvoke(question)
                    all_vector_results.extend(general_results)
                
                # Search without code filters
                no_filter_retriever = chroma.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 10}
                )
                no_filter_results = await no_filter_retriever.ainvoke(question)
                all_vector_results.extend(no_filter_results)
            
            else:
                similarity_search_retriever = chroma.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 20}
                )
                results = await similarity_search_retriever.ainvoke(question)
                all_vector_results.extend(results)

        unique_vector_results = []
        seen_content = set()
        for doc in all_vector_results:
            if doc.page_content not in seen_content:
                unique_vector_results.append(doc)
                seen_content.add(doc.page_content)

        raw_docs = chroma.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=doc, metadata=meta if meta else {})
            for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"] or [{}] * len(raw_docs["documents"]))
        ]
        
        bm25_retriever = BM25Retriever.from_documents(documents=documents, k=20)
        
        similarity_search_retriever = chroma.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 20}
        )
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[similarity_search_retriever, bm25_retriever], 
            weights=[0.5, 0.5]
        )
        
        ensemble_results = await ensemble_retriever.ainvoke(self.user_query)
        
        results_with_urls = []
        for doc in ensemble_results:
            result = {
                'text': doc.page_content,
                'url': doc.metadata.get('url')
            }
            results_with_urls.append(result)
        
        return results_with_urls
    
    async def final_bot(self, results_with_urls: List[Dict[str, Any]]) -> Any:
        """Generate final answer using retrieved information and LLM."""
        
        chunk_texts = [result['text'] for result in results_with_urls]

        system_prompt = f"""
        ### Role
        You are a helpful assistant that answers questions using ONLY the provided information.
        
        ### Problem
        You need to create natural, user-friendly answers from retrieved database content without adding any external knowledge, while filtering out irrelevant information.
        
        ### Task        
        Read all provided information carefully
        Extract ONLY content directly relevant to the user's question
        Ignore information about unrelated topics, languages, or frameworks
        Create a clear, comprehensive answer using only relevant parts
        Preserve exact code snippets if they match the requested language/topic     

        ### Constraints     
        Use ONLY information from the provided content
        Include ONLY information that directly answers the user's question
        If user asks about Python, don't include JavaScript/PHP/etc. examples
        Never add facts or explanations not in the provided text
        If relevant information is incomplete, say "The available information doesn't cover..."     

        ### Input
        Available Information: {chunk_texts}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original query: {self.user_query}")
        ]
        
        response = await self.llm.ainvoke(messages)
        return response

    async def orchestrate_retrieval(self) -> None:
        """Main orchestration method that coordinates the entire retrieval process."""
        
        chroma = self.initialize_chroma_db()
        assistant_response = await self.initial_bot(self.user_query)
        results = await self.information_retrieval(assistant_response, chroma)
        response = await self.final_bot(results)
        print(response.content)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred while processing your request"
        ).dict()
    )

@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest):
    """
    Accept user questions and return RAG-generated answers with source references.
    """
    
    try:
        ir = InformationRetrieval(request.question)
        chroma = ir.initialize_chroma_db()
        assistant_response = await ir.initial_bot(request.question)
        results_with_urls = await ir.information_retrieval(assistant_response, chroma)
        llm_response = await ir.final_bot(results_with_urls)
        
        sources = []
        seen_urls = set()
        for result in results_with_urls:
            url = result.get('url')
            if url and url not in seen_urls:
                sources.append(url)
                seen_urls.add(url)
        
        return QueryResponse(
            answer=llm_response.content,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

@app.get("/stats", response_model=StatsResponse, status_code=status.HTTP_200_OK)
async def stats_endpoint():
    """
    Return comprehensive statistics about the knowledge base.
    """
    
    try:
        chroma_db = ChromaDB(persist_directory="/Users/edvardsivickij/Documents/My documents/oxylabs/VectorDB/chroma_storage")
        collection_data = chroma_db.collection.get(include=["metadatas", "documents"])
        
        total_chunks = len(collection_data.get('ids', []))
        metadatas = collection_data.get('metadatas', [])
        documents = collection_data.get('documents', [])
        
        unique_urls = set()
        sections = {}
        levels = {}
        programming_languages = {
            'curl': 0,
            'python': 0,
            'php': 0,
            'csharp': 0,
            'golang': 0,
            'java': 0,
            'javascript': 0,
            'json': 0,
            'html': 0,
            'sql': 0,
            'general_code': 0
        }
        
        total_characters = 0
        
        for metadata, document in zip(metadatas, documents):
            if metadata:
                # URLs
                if metadata.get('url'):
                    unique_urls.add(metadata['url'])
                
                # Sections
                section_name = metadata.get('section_name', 'Unknown')
                sections[section_name] = sections.get(section_name, 0) + 1
                
                # Levels
                level = metadata.get('level', 'Unknown')
                levels[level] = levels.get(level, 0) + 1
                
                # Programming languages
                for lang in programming_languages.keys():
                    if metadata.get(f'has_{lang}', False):
                        programming_languages[lang] += 1
                if metadata.get('has_code', False):
                    programming_languages['general_code'] += 1
            
            # Document length
            if document:
                total_characters += len(document)
        
        avg_chunk_length = total_characters / total_chunks if total_chunks > 0 else 0
        
        return StatsResponse(
            total_documents=len(unique_urls),
            total_chunks=total_chunks,
            collection_name=chroma_db.collection_name,
            programming_languages=programming_languages,
            sections=sections,
            levels=levels,
            unique_urls=list(unique_urls),
            avg_chunk_length=round(avg_chunk_length, 2),
            total_characters=total_characters,
            last_updated=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch knowledge base statistics: {str(e)}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """

    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoin
    t with API information.
    """
    return {
        "message": "RAG Knowledge Base API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Submit questions for RAG-based answers",
            "GET /stats": "Get knowledge base statistics",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "retrieval_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )