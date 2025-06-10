# ChromaDB
import chromadb

# Formatting
from typing import List, Dict, Any

# Other
import uuid

class ChromaDB:
    def __init__(self, collection_name: str = "oxylabs_docs", persist_directory: str = "./VectorDB/chroma_storage"):
        """Initialize ChromaDB client with collection name and persist directory."""

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, processed_chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """Add processed chunks to ChromaDB collection."""
        
        ids = [str(uuid.uuid4()) for _ in processed_chunks]
        documents = [chunk['text_to_embed'] for chunk in processed_chunks]
        
        metadatas = []
        for chunk in processed_chunks:
            metadata = {
                'section_name': chunk['section_name'],
                'level': str(chunk['level']),
                'url': chunk['url'],
                'has_curl': chunk['has_curl'],
                'has_python': chunk['has_python'],
                'has_php': chunk['has_php'],
                'has_csharp': chunk['has_csharp'],
                'has_golang': chunk['has_golang'],
                'has_java': chunk['has_java'],
                'has_javascript': chunk['has_javascript'],
                'has_json': chunk['has_json'],
                'has_html': chunk['has_html'],
                'has_sql': chunk['has_sql'],
                'has_code': chunk['has_code'],
                'original_text': chunk['text'],
                'full_content': chunk['full_content'],
                'contextualized_content': chunk['contextualized_content'],
            }
            if chunk['parent_url']:
                metadata['parent_url'] = chunk['parent_url']
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return ids