# VectorDB
from VectorDB.chroma_db import ChromaDB

# Langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Formatting
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Other
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

class InformationRetrieval: 
    def __init__(self, user_query: str) -> None:
        """Initialize retrieval system with user query and configure LLM connection."""

        self.chroma_db = ChromaDB(persist_directory="./VectorDB/chroma_storage")
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
    
    def initial_bot(self, user_query: str) -> AssistantResponse:
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
        
        try:
            response = self.llm.invoke(messages)
            response_dict = json.loads(response.content)
            validated_response = AssistantResponse(**response_dict)
            return validated_response
        except Exception as e:
            raise RuntimeError(f"Failed to process query with LLM: {e}")
    
    def information_retrieval(self, assistant_response: AssistantResponse, chroma: Chroma) -> List[Dict[str, Any]]:
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
                    specific_results = specific_retriever.invoke(question)
                    all_vector_results.extend(specific_results)
                
                # Search with general code filter only if no specific results found
                if len(specific_results) == 0:
                    general_retriever = chroma.as_retriever(
                        search_type="similarity",
                        search_kwargs={'k': 10, 'filter': {"has_code": True}}
                    )
                    general_results = general_retriever.invoke(question)
                    all_vector_results.extend(general_results)
                
                # Search without code filters
                no_filter_retriever = chroma.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 10}
                )
                no_filter_results = no_filter_retriever.invoke(question)
                all_vector_results.extend(no_filter_results)
            
            else:
                similarity_search_retriever = chroma.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 20}
                )
                results = similarity_search_retriever.invoke(question)
                all_vector_results.extend(results)

        unique_vector_results = []
        seen_content = set()
        for doc in all_vector_results:
            if doc.page_content not in seen_content:
                unique_vector_results.append(doc)
                seen_content.add(doc.page_content)

        try:
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
            
            ensemble_results = ensemble_retriever.invoke(self.user_query)
            
            results_with_urls = []
            for doc in ensemble_results:
                result = {
                    'text': doc.page_content,
                    'url': doc.metadata.get('url')
                }
                results_with_urls.append(result)
            
            return results_with_urls
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve documents from database: {e}")
    
    def final_bot(self, results_with_urls: List[Dict[str, Any]]) -> Any:
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
        
        response = self.llm.invoke(messages)
        print(response.content)
        return response

    def orchestrate_retrieval(self) -> None:
        """Main orchestration method that coordinates the entire retrieval process."""
        
        chroma = self.initialize_chroma_db()
        assistant_response = self.initial_bot(self.user_query)
        results = self.information_retrieval(assistant_response, chroma)
        self.final_bot(results)


if __name__ == "__main__":
    query = input("Enter your query: ")
    ir = InformationRetrieval(query)
    ir.orchestrate_retrieval()