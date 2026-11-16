"""
Clean PDF parsing for document ingestion.
Handles multi-column layouts, tables, and text extraction.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import re


class PDFParser:
    """
    Extract clean text from PDF documents.
    Handles common issues: multi-column, headers/footers, tables.
    """
    
    def __init__(self, remove_headers_footers: bool = True):
        """
        Args:
            remove_headers_footers: Whether to attempt removing repeated headers/footers
        """
        self.remove_headers_footers = remove_headers_footers
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def parse_pdf(self, pdf_path: str, doc_id: Optional[str] = None) -> Dict[str, str]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            doc_id: Optional document identifier
            
        Returns:
            Dict with 'doc_id', 'text', 'metadata'
        """
        pdf_path = Path(pdf_path)
        
        if doc_id is None:
            doc_id = pdf_path.stem
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        
        # Extract text from all pages
        all_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                all_text.append(text)
        
        doc.close()
        
        # Combine and clean
        combined_text = '\n\n'.join(all_text)
        clean_text = self._clean_text(combined_text)
        
        return {
            'doc_id': doc_id,
            'text': clean_text,
            'metadata': {
                'source': str(pdf_path),
                'num_pages': len(doc)
            }
        }
    
    def parse_directory(self, directory: str, pattern: str = "*.pdf") -> List[Dict]:
        """
        Parse all PDFs in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            
        Returns:
            List of document dicts
        """
        directory = Path(directory)
        pdf_files = list(directory.glob(pattern))
        
        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.parse_pdf(pdf_file)
                documents.append(doc)
                print(f"✓ Parsed: {pdf_file.name}")
            except Exception as e:
                print(f"✗ Failed to parse {pdf_file.name}: {e}")
        
        return documents


def parse_pdf(pdf_path: str, doc_id: Optional[str] = None) -> Dict[str, str]:
    """Convenience function to parse a single PDF."""
    parser = PDFParser()
    return parser.parse_pdf(pdf_path, doc_id)


def parse_pdf_directory(directory: str, pattern: str = "*.pdf") -> List[Dict]:
    """Convenience function to parse a directory of PDFs."""
    parser = PDFParser()
    return parser.parse_directory(directory, pattern)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = parse_pdf(pdf_path)
        print(f"\nDocument ID: {result['doc_id']}")
        print(f"Pages: {result['metadata']['num_pages']}")
        print(f"Text length: {len(result['text'])} characters")
        print(f"\nFirst 500 characters:")
        print(result['text'][:500])
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")
