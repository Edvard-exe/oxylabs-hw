# Core libraries
import ast
import os
import re
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

# Data manipulation
import pandas as pd

# Environment
from dotenv import load_dotenv

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Vectorization:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self._text_splitter = None
        self._compile_language_patterns()

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Text splitter for splitting the content into chunks."""
        
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
                length_function=len,
                is_separator_regex=False,
            )
        return self._text_splitter

    def _compile_language_patterns(self) -> None:
        """Pre-compile regex patterns for efficient language detection."""
        
        self.language_patterns = {
            'has_curl': re.compile(r'curl\s|--user|-h\s|curl_', re.IGNORECASE),
            'has_python': re.compile(r'import\s|def\s|print\(|requests\.', re.IGNORECASE),
            'has_php': re.compile(r'<\?php|\$.*(?:curl_|echo\s)', re.IGNORECASE),
            'has_csharp': re.compile(r'using\s+system|console\.writeline|httpclient', re.IGNORECASE),
            'has_golang': re.compile(r'package\s+main|func\s+main\(\)|fmt\.println|:=', re.IGNORECASE),
            'has_java': re.compile(r'public\s+class|system\.out\.println|import\s+java', re.IGNORECASE),
            'has_javascript': re.compile(r'const\s|console\.log|function\(|require\(', re.IGNORECASE),
            'has_json': re.compile(r'\{".*":', re.IGNORECASE),
            'has_html': re.compile(r'<html>|<!doctype|<div', re.IGNORECASE),
            'has_sql': re.compile(r'select\s+.*from\s+', re.IGNORECASE),
        }

    def data_preparation(self) -> pd.DataFrame:
        """Take content column and flatten it into a single string efficiently."""

        processed_content = []
        for content_str in self.data['content']:
            content = ast.literal_eval(content_str)
            filtered_items = []
            
            for section in content:
                if len(section) > 1:
                    for sublist in section[1:]:
                        for item in sublist:
                            if item and "Last updated" not in item and "Was this helpful?" not in item:
                                filtered_items.append(item)
            
            processed_content.append('\n'.join(filtered_items))

        self.data['content'] = processed_content
        return self.data

    def detect_languages(self, text: str) -> Dict[str, bool]:
        """Detect multiple programming languages in text efficiently using pre-compiled patterns."""
        
        languages = {}
        
        for lang_key, pattern in self.language_patterns.items():
            languages[lang_key] = bool(pattern.search(text))
        
        languages['has_code'] = any(languages.values())
        return languages
    
    def data_splitting(self):
        all_chunks = []
        
        for index, row in self.data.iterrows():
            text = row['content']
            chunks = self.text_splitter.split_text(text)
            
            for chunk in chunks:
                language_flags = self.detect_languages(chunk)
                
                chunk_data = {
                    'text': chunk,
                    'section_name': row['section_name'],
                    'level': row['level'],
                    'url': row['url'],
                    'parent_url': row['parent_url'] if pd.notna(row['parent_url']) else None,
                    'full_content': text,
                    **language_flags  
                }
                all_chunks.append(chunk_data)

        return all_chunks
    
    def add_context(self, chunk: Dict[str, Any]) -> str:
        """
        Add context to a single chunk, by using GPT-4o-mini
        """

        DOCUMENT_CONTEXT = f"""
        <document>
        {chunk['full_content']}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = f"""
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk['text']}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": DOCUMENT_CONTEXT
                    },
                    {
                        "role": "user", 
                        "content": CHUNK_CONTEXT_PROMPT
                    }
                ],
                temperature=0.0,
            )
            context = response.choices[0].message.content
            return context
        except Exception as e:
            return ""

    def process_chunks_parallel(self, chunks: List[Dict[str, Any]], parallel_threads: int = 4) -> List[Dict[str, Any]]:
        """
        Process chunks with context addition in parallel.
        """

        processed_chunks = []
        
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self.add_context, chunk)
                futures.append((future, chunk))
            
            for future, chunk in futures:
                context = future.result()
                chunk['contextualized_content'] = context
                chunk['text_to_embed'] = f"{chunk['text']}\n\n{context}"
                processed_chunks.append(chunk)

        return processed_chunks

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed texts using OpenAI's text-embedding-3-small model.
        """

        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                continue
        
        return all_embeddings

    def process_data(self, parallel_threads: int = 4) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        Complete vectorization pipeline: prepare data, split, add context, and embed.
        Returns processed chunks and their embeddings.
        """
        
        self.data_preparation()
        chunks = self.data_splitting()
        processed_chunks = self.process_chunks_parallel(chunks, parallel_threads)
        
        texts_to_embed = [chunk['text_to_embed'] for chunk in processed_chunks]
        embeddings = self.embed_texts(texts_to_embed)
        
        return processed_chunks, embeddings
