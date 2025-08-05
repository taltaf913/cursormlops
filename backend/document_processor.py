import os
import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.processed_documents = []
    
    def process_document(self, content: str, filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Process a document and return a list of Document objects
        """
        try:
            # Detect file type and process accordingly
            file_extension = self._get_file_extension(filename)
            
            if file_extension in ['.txt', '.md']:
                processed_content = self._process_text_file(content)
            elif file_extension == '.html':
                processed_content = self._process_html_file(content)
            else:
                # Default to text processing
                processed_content = self._process_text_file(content)
            
            # Create document with metadata
            document = Document(
                page_content=processed_content,
                metadata={
                    "filename": filename,
                    "file_type": file_extension,
                    "source": filename,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            self.processed_documents.extend(chunks)
            logger.info(f"Processed {filename} into {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        return os.path.splitext(filename.lower())[1]
    
    def _process_text_file(self, content: str) -> str:
        """Process plain text files"""
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        return content
    
    def _process_html_file(self, content: str) -> str:
        """Process HTML files by extracting text content"""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.warning(f"Error parsing HTML, treating as plain text: {e}")
            return self._process_text_file(content)
    
    def _process_markdown_file(self, content: str) -> str:
        """Process Markdown files"""
        import markdown
        
        try:
            # Convert markdown to HTML first
            html = markdown.markdown(content)
            # Then extract text from HTML
            return self._process_html_file(html)
        except Exception as e:
            logger.warning(f"Error parsing Markdown, treating as plain text: {e}")
            return self._process_text_file(content)
    
    def get_processed_documents(self) -> List[Document]:
        """Get all processed documents"""
        return self.processed_documents
    
    def clear_processed_documents(self) -> None:
        """Clear the list of processed documents"""
        self.processed_documents = []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not self.processed_documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "file_types": {}
            }
        
        file_types = {}
        for doc in self.processed_documents:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            "total_documents": len(set(doc.metadata.get("filename") for doc in self.processed_documents)),
            "total_chunks": len(self.processed_documents),
            "file_types": file_types
        }