# Core libraries
import os
from typing import List, Tuple

# Data manipulation
import pandas as pd

# Web scraping
import requests
from lxml import html

# Environment
import dotenv

dotenv.load_dotenv()

USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')

class Scraper:
    def __init__(self, url: str, section_name: str) -> None:
        self.url = url
        self.section_name = section_name
        self._session = None

    @property
    def session(self) -> requests.Session:
        """Requests session for connection reuse."""
        
        if self._session is None:
            self._session = requests.Session()
            self._session.auth = (USERNAME, PASSWORD)
        return self._session

    def get_html(self, url: str) -> str:
        """Get the HTML content of a given URL efficiently."""
        
        payload = {
            'source': 'universal',
            'url': url,
            'render': 'html',
        }
        
        response = requests.request('POST', 'https://realtime.oxylabs.io/v1/queries', auth=(USERNAME, PASSWORD), json=payload)
        html = response.json().get('results', [{}])[0].get('content', '')
        
        return html
    
    def get_urls(self, html_content: str) -> list[str]:
        """
        Get the URLs from the sidebar of the given HTML content.
        """

        tree = html.fromstring(html_content)

        # Find the sidebar navigation
        sidebar = tree.xpath('//aside[@data-testid="table-of-contents"]')[0]

        urls = []
        section_found = False
        section_base_path = None
        
        links = sidebar.xpath('.//a[@href]')

        for link in links:
            href = link.get('href')
            text = link.text_content().strip()
        
            if text == self.section_name:
                section_found = True
                section_base_path = href
                urls.append(href)
                continue
        
            # If we found target section, collect all subsequent links that belong to it
            if section_found and section_base_path and href.startswith(section_base_path):
                urls.append(href)
            
            # Stop collecting when we reach another main section
            elif section_found and href and section_base_path and not href.startswith(section_base_path):
                # Check if this is a different main section
                if not href.startswith('#') and len(href.split('/')) <= len(section_base_path.split('/')):
                    break

        return urls

    def parse_html(self, html_content: str) -> List[Tuple[str, List[str], List[str]]]:
        """Parse the HTML content and return structured data efficiently."""
        
        tree = html.fromstring(html_content)
        main = tree.xpath('//main')[0]

        result = []
        current_header = None
        current_paragraphs = []
        current_code_blocks = []
        
        for elem in main.iterdescendants():
            if elem.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_header:
                    result.append((current_header, current_paragraphs, current_code_blocks))
                current_header = elem.text_content().strip()
                current_paragraphs = []
                current_code_blocks = []
            elif elem.tag == 'p' and current_header:
                text = elem.text_content().strip()
                if text:
                    current_paragraphs.append(text)
            elif elem.tag == 'pre' and current_header:
                code_elem = elem.find('.//code')
                if code_elem is not None:
                    code_text = code_elem.text_content().strip()
                    if code_text:
                        current_code_blocks.append(code_text)
        
        if current_header:
            result.append((current_header, current_paragraphs, current_code_blocks))
        
        return result

    def get_data(self) -> pd.DataFrame:
        """Get the data from the given URL efficiently."""
        
        original_html = self.get_html(self.url)
        parsed_original_html = self.parse_html(original_html)

        urls = self.get_urls(original_html)
        urls = urls[1:] 

        data_rows = [{
            'url': self.url,
            'content': parsed_original_html,
            'level': 0,
            'parent_url': None,
            'section_name': self.section_name
        }]

        # Process URLs efficiently
        for url in urls:
            full_url = f"https://developers.oxylabs.io{url}"
            print(full_url)
            
            url_parts = url.strip('/').split('/')
            level = len(url_parts) - 2
            
            parent_url = (
                self.url if level == 1 
                else f"https://developers.oxylabs.io/{'/'.join(url_parts[:-1])}" if level > 1 
                else None
            )
            section_name = url_parts[-1].replace('-', ' ').title()
            
            html_content = self.get_html(full_url)
            parsed_content = self.parse_html(html_content)
            
            data_rows.append({
                'url': full_url,
                'content': parsed_content,
                'level': level,
                'parent_url': parent_url,
                'section_name': section_name
            })

        return pd.DataFrame(data_rows)