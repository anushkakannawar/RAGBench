"""
Reproducible document chunking with configurable size and overlap.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    metadata: Optional[Dict] = None


class DocumentChunker:
    """
    Chunks documents with fixed token size and overlap.
    Uses simple whitespace tokenization for reproducibility.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        """
        Args:
            chunk_size: Target number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.split()
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        return " ".join(tokens)
    
    def chunk_document(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Split document into overlapping chunks.
        
        Args:
            text: Document text to chunk
            doc_id: Unique document identifier
            
        Returns:
            List of Chunk objects
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return []
        
        tokens = self._simple_tokenize(text)
        chunks = []
        
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            # Extract chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self._detokenize(chunk_tokens)
            
            # Calculate character positions (approximate)
            char_start = len(self._detokenize(tokens[:start_idx]))
            char_end = char_start + len(chunk_text)
            
            # Create chunk
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_chunk_{chunk_num}",
                doc_id=doc_id,
                start_char=char_start,
                end_char=char_end,
                metadata={
                    "token_count": len(chunk_tokens),
                    "chunk_index": chunk_num
                }
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.overlap
            chunk_num += 1
            
            # Break if we've reached the end
            if end_idx >= len(tokens):
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of dicts with 'doc_id' and 'text' keys
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            doc_id = doc.get('doc_id')
            text = doc.get('text', '')
            
            if not doc_id:
                raise ValueError("Each document must have a 'doc_id' field")
            
            chunks = self.chunk_document(text, doc_id)
            all_chunks.extend(chunks)
        
        return all_chunks


def create_chunker(chunk_size: int = 512, overlap: int = 64) -> DocumentChunker:
    """Factory function to create a chunker with specified parameters."""
    return DocumentChunker(chunk_size=chunk_size, overlap=overlap)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
    with language generation. The system first retrieves relevant documents from a knowledge 
    base, then uses those documents as context for generating responses. This approach helps 
    reduce hallucinations and grounds the model's outputs in factual information.
    
    However, RAG systems can still fail in several ways. Chunk boundary fragmentation occurs 
    when relevant information is split across multiple chunks. Multi-hop reasoning failures 
    happen when the question requires combining facts from different sources.
    """ * 50  # Repeat to create longer text
    
    chunker = create_chunker(chunk_size=128, overlap=16)
    chunks = chunker.chunk_document(sample_text, doc_id="sample_doc")
    
    print(f"Created {len(chunks)} chunks")
    print(f"\nFirst chunk:")
    print(f"  ID: {chunks[0].chunk_id}")
    print(f"  Tokens: {chunks[0].metadata['token_count']}")
    print(f"  Text preview: {chunks[0].text[:100]}...")
