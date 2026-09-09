import logfire

def parse_text(file_path: str):
    """
    Parses plain textfiles.
    """
    with logfire.span("Text Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content

        except Exception as e:
            logfire.error(f"Error parsing text file {file_path}: {e}")
            raise e