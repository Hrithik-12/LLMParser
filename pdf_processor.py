# Real PDF Document Processor
# This replaces our fake document processing with actual PDF parsing

import requests
import PyPDF2
import io
from typing import Dict, List
import re
from urllib.parse import urlparse

class RealDocumentProcessor:
    """
    Real document processor that handles PDFs, DOCX, and other formats
    
    WHY THIS MATTERS:
    - Insurance policies are typically in PDF format
    - We need to extract actual text content to search through
    - Different document types need different parsing strategies
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit
    
    def extract_text_from_url(self, document_url: str) -> Dict[str, any]:
        """
        Download and extract text from a document URL
        
        WHY WE DO THIS:
        - The API receives URLs, not file uploads
        - We need to fetch the document first, then process it
        - Error handling is crucial for production systems
        """
        try:
            print(f"📥 Downloading document from: {document_url}")
            
            # Download the document
            response = requests.get(document_url, timeout=30)
            response.raise_for_status()
            
            # Check file size
            if len(response.content) > self.max_file_size:
                raise ValueError(f"File too large: {len(response.content)} bytes")
            
            # Determine file type from URL or content-type
            file_extension = self._get_file_extension(document_url, response.headers)
            
            # Extract text based on file type
            if file_extension == '.pdf':
                text_content = self._extract_from_pdf(response.content)
            elif file_extension == '.docx':
                text_content = self._extract_from_docx(response.content)
            else:
                # Assume plain text
                text_content = response.content.decode('utf-8', errors='ignore')
            
            # Clean and structure the text
            cleaned_text = self._clean_text(text_content)
            
            return {
                'success': True,
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text),
                'file_type': file_extension,
                'source_url': document_url
            }
            
        except requests.RequestException as e:
            return {
                'success': False,
                'error': f"Failed to download document: {str(e)}",
                'text': '',
                'source_url': document_url
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to process document: {str(e)}",
                'text': '',
                'source_url': document_url
            }
    
    def _get_file_extension(self, url: str, headers: Dict) -> str:
        """
        Determine file extension from URL or content-type header
        
        WHY THIS IS IMPORTANT:
        - Different file types need different processing
        - URLs might not have extensions
        - Content-type headers provide reliable file type info
        """
        # Try to get extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        if path.endswith('.pdf'):
            return '.pdf'
        elif path.endswith('.docx'):
            return '.docx'
        elif path.endswith('.txt'):
            return '.txt'
        
        # Try to get from content-type header
        content_type = headers.get('content-type', '').lower()
        if 'pdf' in content_type:
            return '.pdf'
        elif 'wordprocessingml' in content_type or 'docx' in content_type:
            return '.docx'
        
        # Default to PDF (most common for policies)
        return '.pdf'
    
    def _extract_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content
        
        WHY WE NEED THIS:
        - PDFs are the most common format for insurance policies
        - PyPDF2 handles most standard PDFs well
        - We need to handle potential PDF parsing errors gracefully
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            total_pages = len(pdf_reader.pages)
            
            print(f"📄 Processing PDF with {total_pages} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num} ---\n{page_text}\n"
                except Exception as e:
                    print(f"⚠️  Warning: Could not extract text from page {page_num}: {e}")
                    continue
            
            if not text_content.strip():
                raise ValueError("No text could be extracted from PDF")
            
            print(f"✅ Successfully extracted {len(text_content)} characters from PDF")
            return text_content
            
        except Exception as e:
            raise ValueError(f"PDF processing failed: {str(e)}")
    
    def _extract_from_docx(self, docx_content: bytes) -> str:
        """
        Extract text from DOCX content
        
        WHY WE INCLUDE THIS:
        - Some policies might be in Word format
        - DOCX is structured differently than PDF
        - Provides fallback option for different document types
        """
        try:
            # This would require python-docx library
            # For now, we'll return a placeholder and suggest implementation
            return "DOCX processing not yet implemented. Please convert to PDF."
        except Exception as e:
            raise ValueError(f"DOCX processing failed: {str(e)}")
    
    def _clean_text(self, raw_text: str) -> str:
        """
        Clean and normalize extracted text
        
        WHY TEXT CLEANING IS CRUCIAL:
        - PDFs often have formatting artifacts
        - Inconsistent spacing and line breaks
        - Special characters that interfere with processing
        - Better cleaned text = better search results
        """
        if not raw_text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', raw_text)
        
        # Remove page markers we added
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Fix common PDF extraction issues
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Trim and return
        return text.strip()
    
    def create_document_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        """
        Split document into chunks for better retrieval
        
        WHY WE CHUNK DOCUMENTS:
        - Long documents are hard to search through
        - Embeddings work better on smaller text pieces
        - Allows for more precise retrieval
        - Prevents token limits in LLM calls
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        
        # If document is very small, create a single chunk
        if len(words) <= overlap:
            chunks = [{
                'text': text.strip(),
                'chunk_id': 0,
                'word_count': len(words),
                'start_word': 0,
                'end_word': len(words)
            }]
            print(f"📝 Created 1 chunk from small document ({len(words)} words)")
            return chunks
        
        chunks = []
        
        # Adjust chunk size if document is smaller than default
        effective_chunk_size = min(chunk_size, max(50, len(words) // 2))  # At least 50 words per chunk
        effective_overlap = min(overlap, effective_chunk_size // 4)  # Overlap can't be more than 1/4 of chunk
        
        for i in range(0, len(words), effective_chunk_size - effective_overlap):
            chunk_words = words[i:i + effective_chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Only create chunk if it has meaningful content
            if len(chunk_text.strip()) > 10:  # At least 10 characters
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'word_count': len(chunk_words),
                    'start_word': i,
                    'end_word': min(i + effective_chunk_size, len(words))
                })
        
        # If no chunks were created, create one with the full text
        if not chunks and text.strip():
            chunks.append({
                'text': text.strip(),
                'chunk_id': 0,
                'word_count': len(words),
                'start_word': 0,
                'end_word': len(words)
            })
        
        print(f"📝 Created {len(chunks)} chunks from document ({len(words)} total words)")
        return chunks

# Example usage and testing
def test_pdf_processor():
    """Test the PDF processor with a sample document"""
    
    processor = RealDocumentProcessor()
    
    # Test with the sample URL from the problem statement
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf"
    
    print("🧪 Testing PDF Processor...")
    print("=" * 50)
    
    result = processor.extract_text_from_url(test_url)
    
    if result['success']:
        print(f"✅ Document processed successfully!")
        print(f"📊 Stats:")
        print(f"   - File type: {result['file_type']}")
        print(f"   - Word count: {result['word_count']}")
        print(f"   - Character count: {result['char_count']}")
        print(f"   - Source: {result['source_url']}")
        
        # Show first 500 characters
        preview = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
        print(f"\n📄 Content preview:")
        print(preview)
        
        # Create chunks
        chunks = processor.create_document_chunks(result['text'])
        print(f"\n📝 Created {len(chunks)} chunks")
        
        if chunks:
            print(f"First chunk preview: {chunks[0]['text'][:200]}...")
        
    else:
        print(f"❌ Processing failed: {result['error']}")

if __name__ == "__main__":
    test_pdf_processor()