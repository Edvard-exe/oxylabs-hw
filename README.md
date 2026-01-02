# Oxylabs Documentation Assistant

A comprehensive RAG (Retrieval-Augmented Generation) system for Oxylabs documentation that combines web scraping, vector databases, and AI to provide intelligent question-answering capabilities with source attribution.

## Project Overview

This project implements a sophisticated documentation assistant that scrapes Oxylabs developer documentation, processes it into a vector database, and provides multiple interfaces (CLI, REST API, Web UI) for intelligent question answering with accurate source references.

### Key Features

- **Intelligent Web Scraping**: Automated scraping of Oxylabs documentation with hierarchical content extraction
- **Advanced Vector Database**: ChromaDB-powered semantic search with programming language detection
- **Multi-Modal Retrieval**: Combines vector similarity search with BM25 for optimal results
- **Multiple Interfaces**: CLI script, FastAPI REST API, and Streamlit web interface
- **Source Attribution**: All answers include direct links to source documentation
- **Programming Language Awareness**: Detects and filters content by programming languages (Python, PHP, cURL, JavaScript, etc.)

## Project Structure

```
oxylabs/
├── FastAPI/              # REST API implementation
├── Streamlit/            # Web interface
├── VectorDB/            # Data processing and vector database
├── information_retrieval.py  # Core CLI interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Oxylabs credentials (for scraping)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd oxylabs
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   USERNAME=your_oxylabs_username
   PASSWORD=your_oxylabs_password
   ```

### Basic Usage

**Option 1: CLI Interface (Quickest)**
```bash
python information_retrieval.py
# Enter your question when prompted
```

**Option 2: Web Interface (Most User-Friendly)**
```bash
streamlit run Streamlit/frontend.py
# Open browser to http://localhost:8501
```

**Option 3: REST API (For Integrations)**
```bash
uvicorn FastAPI.retrieval_api:app --reload
# API available at http://localhost:8000
```

## VectorDB/ - Data Processing Pipeline

The `VectorDB/` folder contains the core data processing pipeline that scrapes, processes, and stores documentation in a vector database.

### Components

#### `main.py` - Orchestration Engine
**Purpose**: Coordinates the entire data processing workflow.

**Key Features**:
- URL and section management
- Data scraping orchestration
- Vector database population
- CSV data persistence

**Usage**:
```python
from VectorDB.main import Main

# Initialize with target URLs and sections
urls = [
    "https://developers.oxylabs.io/scraping-solutions/web-scraper-api",
    "https://developers.oxylabs.io/proxies/mobile-proxies"
]
sections = ["Web Scraper API", "Mobile Proxies"]

main = Main(urls, sections)

# Fresh scraping and processing
data = main.run()

# Or load from existing CSV
data = main.run_from_csv('scraped_data.csv')
```

#### `scraper.py` - Intelligent Web Scraper
**Purpose**: Extracts structured content from Oxylabs documentation pages.

**Key Features**:
- Hierarchical URL discovery from navigation sidebars
- Structured content extraction (headers, paragraphs, code blocks)
- Parent-child URL relationship mapping
- Efficient session management with authentication

**How it works**:
1. Fetches HTML content using Oxylabs Real-time API
2. Parses navigation sidebar to discover related URLs
3. Extracts structured content (headers, text, code) from each page
4. Maintains hierarchical relationships between pages

**Output**: DataFrame with columns:
- `url`: Page URL
- `content`: Structured content (headers, paragraphs, code blocks)
- `level`: Hierarchy level (0=main page, 1=subsection, etc.)
- `parent_url`: Parent page URL
- `section_name`: Section identifier

#### `vectorization.py` - Text Processing & Embedding
**Purpose**: Converts raw scraped content into searchable vector embeddings.

**Key Features**:
- Intelligent text chunking with overlap
- Programming language detection (10+ languages)
- Contextual enhancement using GPT-4o-mini
- Parallel processing for efficiency
- OpenAI embeddings generation

**Processing Pipeline**:
1. **Content Preparation**: Flattens nested content structures
2. **Text Splitting**: Creates optimal chunks (2000 chars, 200 overlap)
3. **Language Detection**: Identifies programming languages in each chunk
4. **Context Enhancement**: Adds contextual information using AI
5. **Embedding Generation**: Creates vector embeddings using OpenAI

**Language Detection Supports**:
- cURL, Python, PHP, C#, Go, Java, JavaScript
- JSON, HTML, SQL
- General code detection

#### `chroma_db.py` - Vector Database Interface
**Purpose**: Manages ChromaDB operations for vector storage and retrieval.

**Key Features**:
- Persistent vector storage
- Metadata preservation
- Efficient batch operations
- Collection management

**Usage**:
```python
from VectorDB.chroma_db import ChromaDB

# Initialize database
chroma_db = ChromaDB(persist_directory="./VectorDB/chroma_storage")

# Add processed documents
chroma_db.add_documents(processed_chunks, embeddings)
```

### Running VectorDB Pipeline

**Complete Pipeline (Fresh Scraping)**:
```bash
cd VectorDB/
python main.py
```

**From Existing Data**:
```python
from VectorDB.main import Main

main = Main(urls=[], sections=[])
main.run_from_csv('scraped_data.csv')
```

## FastAPI/ - REST API Service

The `FastAPI/` folder provides a production-ready REST API for the documentation assistant.

### Components

#### `retrieval_api.py` - Main API Service
**Purpose**: Provides HTTP endpoints for question answering and system statistics.

**Key Features**:
- Async/await support for high performance
- Comprehensive error handling
- CORS support for web applications
- Request/response validation with Pydantic
- Detailed API documentation (auto-generated)

### API Endpoints

#### `POST /query` - Ask Questions
**Purpose**: Submit questions and receive AI-generated answers with sources.

**Request Format**:
```json
{
  "question": "How do I use Python with Oxylabs API?"
}
```

**Response Format**:
```json
{
  "answer": "To use Python with Oxylabs API, you can...",
  "sources": [
    "https://developers.oxylabs.io/scraping-solutions/web-scraper-api/python",
    "https://developers.oxylabs.io/scraping-solutions/web-scraper-api/getting-started"
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### `GET /stats` - System Statistics
**Purpose**: Retrieve detailed information about the knowledge base.

**Response includes**:
- Total documents and chunks
- Programming language distribution
- Section breakdown
- Average chunk length
- Unique URLs count

#### `GET /health` - Health Check
**Purpose**: Verify API service status.

#### `GET /` - API Information
**Purpose**: Basic API information and version.

### Running FastAPI Service

**Development Mode**:
```bash
cd FastAPI/
uvicorn retrieval_api:app --reload --host 0.0.0.0 --port 8000
```

**Production Mode**:
```bash
cd FastAPI/
uvicorn retrieval_api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Access Points**:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- OpenAPI Schema: http://localhost:8000/openapi.json

### Testing the API

#### `test_api.py` - Comprehensive Test Suite
**Purpose**: Validates API functionality and performance.

**Run Tests**:
```bash
cd FastAPI/
python test_api.py
```

**Test Coverage**:
- Query endpoint validation
- Error handling
- Response format verification
- Statistics endpoint testing
- Health check verification

## 💻 Streamlit/ - Web Interface

The `Streamlit/` folder provides an intuitive web interface for end users.

### Components

#### `frontend.py` - User Interface
**Purpose**: Creates an interactive web application for documentation queries.

**Key Features**:
- Clean, responsive design
- Real-time chat interface
- Source attribution sidebar
- Session state management
- Loading indicators

**Interface Elements**:
- **Banner**: Welcome message and usage instructions
- **Chat Interface**: Conversational question-answering
- **Sidebar**: Live source references
- **Clear Chat**: Reset conversation history

#### `backend.py` - Processing Logic
**Purpose**: Handles question processing and response generation for the web interface.

**Features**:
- Streamlit-optimized processing
- Response caching
- Error handling
- Source URL extraction

### Running Streamlit Application

**Start the Web App**:
```bash
streamlit run Streamlit/frontend.py
```

**Configuration Options**:
```bash
# Custom port
streamlit run Streamlit/frontend.py --server.port 8501

# Different host
streamlit run Streamlit/frontend.py --server.address 0.0.0.0
```

**Access**: http://localhost:8501

### User Experience

1. **Welcome Screen**: Introduction and usage instructions
2. **Question Input**: Natural language question entry
3. **Processing**: Visual loading indicator during search
4. **Response Display**: AI-generated answer with formatting
5. **Source References**: Clickable links to original documentation
6. **Chat History**: Persistent conversation within session

## 🔧 information_retrieval.py - Core CLI Interface

The main CLI script provides direct access to the retrieval system.

### Features

**Intelligent Query Processing**:
- Transforms single questions into multiple search variations
- Detects programming language context
- Optimizes retrieval strategy based on query type

**Advanced Retrieval System**:
- **Vector Search**: Semantic similarity using OpenAI embeddings
- **BM25 Search**: Keyword-based retrieval
- **Ensemble Method**: Combines both approaches for optimal results
- **Language Filtering**: Filters results by detected programming languages

**Response Generation**:
- Uses GPT-4.1 for coherent answer synthesis
- Includes direct source citations
- Maintains context across conversation

### Usage

**Interactive Mode**:
```bash
python information_retrieval.py
# Follow prompts to enter questions
```

**Programmatic Usage**:
```python
from information_retrieval import InformationRetrieval

# Initialize with question
ir = InformationRetrieval("How do I use Python with Oxylabs?")

# Get response
response = ir.orchestrate_retrieval()
print(response)
```

### Processing Pipeline

1. **Query Analysis**: Transforms input into optimized search queries
2. **Language Detection**: Identifies programming language context
3. **Vector Retrieval**: Searches vector database with multiple strategies
4. **BM25 Retrieval**: Keyword-based search for completeness
5. **Result Fusion**: Combines and deduplicates results
6. **Answer Generation**: Creates coherent response with sources


## Current Issues & Limitations

### Known Issues

1. **Suboptimal Document Ranking** - Retrieved documents are not always ranked by relevance
2. **Limited Metadata Usage** - Document metadata is underutilized for filtering and ranking
3. **Inefficient Content Scraping** - Scraping includes unnecessary content like video descriptions and navigation elements
4. **Poor Table Vectorization** - Tables and complex formatting lose structure during processing
5. **ChromaDB Performance Issues** - Requires separate BM25 implementation and loading all documents
6. **Hardcoded API Keys** - OpenAI API keys are fixed in environment variables
7. **Excessive Data Retrieval** - Sometimes retrieves too much irrelevant data
8. **Missing Evaluation Framework** - No systematic evaluation of retrieval and generation quality
9. **Inefficient BM25 Implementation** - Loads entire document collection for BM25 search
