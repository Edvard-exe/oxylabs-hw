# VectorDB modules
from scraper import Scraper
from vectorization import Vectorization
from chroma_db import ChromaDB

# Core libraries
from typing import List, Dict, Any
import pandas as pd

class Main:
    def __init__(self, urls: List[str], sections: List[str]) -> None:
        """Initialize Main class with URLs and sections."""
        
        self.urls = urls
        self.sections = sections
        self._chroma_db = None  

    @property
    def chroma_db(self) -> ChromaDB:
        """Initialize ChromaDB client."""
        
        if self._chroma_db is None:
            self._chroma_db = ChromaDB()
        return self._chroma_db

    def scrape_data(self) -> pd.DataFrame:
        """Scrape data from URLs and sections efficiently using list comprehension."""
   
        scraped_data = [
            Scraper(url, section).get_data() 
            for url, section in zip(self.urls, self.sections)
        ]
        
        final_data = pd.concat(scraped_data, ignore_index=True)
        final_data.to_csv('scraped_data.csv', index=False)
        return final_data
    
    def populate_vector_db(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Populate vector database with processed chunks efficiently."""
        
        vectorization = Vectorization(data)
        processed_chunks, embeddings = vectorization.process_data()
        
        self.chroma_db.add_documents(processed_chunks, embeddings)
        return processed_chunks

    def run(self) -> pd.DataFrame:
        """Run the main workflow efficiently."""
        
        data = self.scrape_data()
        self.populate_vector_db(data)
        return data

    def run_from_csv(self, csv_path: str = 'scraped_data.csv') -> pd.DataFrame:
        """Run the main workflow from a CSV file efficiently."""
        
        data = pd.read_csv(csv_path)
        self.populate_vector_db(data)
        return data


if __name__ == "__main__":
    urls = [
        "https://developers.oxylabs.io/scraping-solutions/web-scraper-api", 
        "https://developers.oxylabs.io/proxies/mobile-proxies"
    ]
    sections = ["Web Scraper API", "Mobile Proxies"]

    main = Main(urls, sections)
    main.run_from_csv()