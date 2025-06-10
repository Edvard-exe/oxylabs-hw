# Testing
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# FastAPI
from fastapi.testclient import TestClient

# Retrieval API
from retrieval_api import app, InformationRetrieval, QueryRequest, AssistantResponse

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"

def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "RAG Knowledge Base API"
    assert data["version"] == "1.0.0"

def test_query_request_validation():
    """Test QueryRequest model validation."""

    request = QueryRequest(question="What is Python?")
    assert request.question == "What is Python?"

    with pytest.raises(ValueError):
        QueryRequest(question="")

@patch('retrieval_api.InformationRetrieval')
def test_query_endpoint_success(mock_ir_class):
    """Test successful query endpoint."""

    mock_ir = Mock()
    mock_ir_class.return_value = mock_ir
    
    mock_ir.initialize_chroma_db.return_value = Mock()
    
    mock_assistant_response = AssistantResponse(
        questions=["What is Python?"],
        has_code=True,
        has_python=True
    )
    mock_ir.initial_bot = AsyncMock(return_value=mock_assistant_response)
    
    mock_results = [{'text': 'Python is a programming language', 'url': 'https://python.org'}]
    mock_ir.information_retrieval = AsyncMock(return_value=mock_results)
    
    mock_llm_response = Mock()
    mock_llm_response.content = "Python is a high-level programming language."
    mock_ir.final_bot = AsyncMock(return_value=mock_llm_response)
    
    response = client.post("/query", json={"question": "What is Python?"})
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["sources"] == ["https://python.org"]

def test_query_endpoint_validation_error():
    """Test query endpoint with validation error."""

    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422

@patch('retrieval_api.ChromaDB')
def test_stats_endpoint_success(mock_chroma_class):
    """Test successful stats endpoint."""

    mock_chroma = Mock()
    mock_chroma_class.return_value = mock_chroma
    mock_chroma.collection_name = "test_collection"
    
    mock_collection = Mock()
    mock_chroma.collection = mock_collection
    
    mock_collection.get.return_value = {
        'ids': ['1', '2'],
        'metadatas': [
            {'url': 'https://example.com', 'section_name': 'intro', 'level': 'basic', 'has_python': True, 'has_code': True},
            {'url': 'https://example.com', 'section_name': 'advanced', 'level': 'expert', 'has_java': True, 'has_code': True}
        ],
        'documents': ['Python content', 'Java content']
    }
    
    response = client.get("/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] == 1  # Unique URLs
    assert data["total_chunks"] == 2
    assert data["collection_name"] == "test_collection"

@patch('retrieval_api.ChromaDB')
@patch('retrieval_api.ChatOpenAI')
def test_information_retrieval_initial_bot(mock_llm, mock_chroma):
    """Test InformationRetrieval initial_bot method."""

    mock_llm.return_value = Mock()
    ir = InformationRetrieval("test query")
    
    mock_response = Mock()
    mock_response.content = '{"questions": ["What is Python?"], "has_code": true, "has_python": true, "has_curl": false, "has_php": false, "has_csharp": false, "has_golang": false, "has_java": false, "has_javascript": false, "has_json": false, "has_html": false, "has_sql": false}'
    ir.llm.ainvoke = AsyncMock(return_value=mock_response)
    
    result = asyncio.run(ir.initial_bot("What is Python?"))
    
    assert isinstance(result, AssistantResponse)
    assert len(result.questions) == 1
    assert result.has_python is True 