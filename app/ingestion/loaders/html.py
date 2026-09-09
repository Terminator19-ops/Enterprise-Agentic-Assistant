from bs4 import BeautifulSoup
import logfire

def parse_html(file_path: str):
    """
    Parses the given HTML content using BeautifulSoup.
    cleans scripts, styles and extracts readable text for RAG
    """
    with logfire.span("HTML Parsing",filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style','meta','noscript']):
                script.decompose()

            text = soup.get_text(separator='\n')

            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)
            return text_clean

        
        except Exception as e:
            
            logfire.error(f"Error parsing HTML file {file_path}: {e}")
            raise e