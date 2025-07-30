# Complete Integrated LLM Query-Retrieval System
# This brings together all components into a production-ready system

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom components
# (In practice, these would be in separate files)
from pdf_processor import RealDocumentProcessor
from vector_search import DocumentSearchSystem
from llm_processor import EnhancedQueryProcessor

class CompleteQueryRetrievalSystem:
    """
    Complete system that integrates all components
    
    WHY WE CREATE A UNIFIED SYSTEM:
    - Single point of entry for all operations
    - Manages the complete pipeline from URL to answers
    - Handles caching and optimization
    - Provides consistent error handling
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize all system components"""
        print("🚀 Initializing Complete Query-Retrieval System...")
        
        # Initialize components
        self.pdf_processor = RealDocumentProcessor()
        self.search_system = DocumentSearchSystem()
        self.query_processor = EnhancedQueryProcessor(gemini_api_key)
        
        # System state
        self.current_document = None
        self.is_ready = False
        self.processing_stats = {
            'documents_processed': 0,
            'queries_handled': 0,
            'total_processing_time': 0,
            'last_document_url': None
        }
        
        print("✅ System initialization complete!")
    
    def process_document_and_queries(self, document_url: str, questions: List[str]) -> Dict:
        """
        Main method that processes a document and answers questions
        
        THIS IS THE CORE WORKFLOW:
        1. Check if we need to process a new document
        2. Download and parse the PDF
        3. Create vector embeddings and search index
        4. For each question:
           - Find relevant document sections using vector search
           - Use LLM to generate intelligent answers
        5. Return formatted results
        """
        start_time = time.time()
        
        try:
            print(f"📋 Processing request:")
            print(f"   Document: {document_url}")
            print(f"   Questions: {len(questions)}")
            
            # Step 1: Check if we need to process the document
            if not self._is_document_ready(document_url):
                print("📥 New document detected, processing...")
                success = self._process_new_document(document_url)
                if not success:
                    raise Exception("Document processing failed")
            else:
                print("✅ Document already processed, using cached version")
            
            # Step 2: Process all questions
            print("🔍 Processing questions...")
            answers = []
            
            for i, question in enumerate(questions, 1):
                print(f"   Question {i}/{len(questions)}: {question[:50]}...")
                
                try:
                    # Find relevant document sections
                    relevant_chunks = self.search_system.search_document(question, top_k=3)
                    
                    # Generate answer using LLM
                    result = self.query_processor.process_query_with_context(question, relevant_chunks)
                    answers.append(result['formatted_answer'])
                    
                except Exception as e:
                    print(f"❌ Failed to process question {i}: {e}")
                    answers.append(f"Unable to process this question: {str(e)}")
            
            # Step 3: Update statistics and return results
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(questions))
            
            result = {
                'answers': answers,
                'processing_time_seconds': round(processing_time, 2),
                'document_processed': document_url,
                'questions_count': len(questions),
                'system_ready': self.is_ready
            }
            
            print(f"✅ Request completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"System error: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                'answers': [error_msg for _ in questions],
                'processing_time_seconds': round(time.time() - start_time, 2),
                'document_processed': document_url,
                'questions_count': len(questions),
                'system_ready': False,
                'error': error_msg
            }
    
    def _is_document_ready(self, document_url: str) -> bool:
        """
        Check if the document is already processed and ready for queries
        
        WHY WE CHECK THIS:
        - Avoid reprocessing the same document
        - Save time and API costs
        - Better user experience with faster responses
        """
        return (self.is_ready and 
                self.current_document == document_url and 
                self.search_system.is_ready)
    
    def _process_new_document(self, document_url: str) -> bool:
        """
        Process a new document and prepare it for queries
        
        WHY THIS IS A SEPARATE METHOD:
        - Document processing is expensive and slow
        - Need robust error handling
        - Want to track processing stats
        """
        try:
            # Process the document and build search index
            success = self.search_system.process_document(document_url, self.pdf_processor)
            
            if success:
                self.current_document = document_url
                self.is_ready = True
                self.processing_stats['documents_processed'] += 1
                self.processing_stats['last_document_url'] = document_url
                print("✅ Document processing and indexing complete")
                return True
            else:
                print("❌ Document processing failed")
                return False
                
        except Exception as e:
            print(f"❌ Document processing error: {e}")
            self.is_ready = False
            return False
    
    def _update_stats(self, processing_time: float, question_count: int):
        """Update system statistics"""
        self.processing_stats['queries_handled'] += question_count
        self.processing_stats['total_processing_time'] += processing_time
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics"""
        return {
            'system_ready': self.is_ready,
            'current_document': self.current_document,
            'stats': self.processing_stats.copy(),
            'average_processing_time': (
                self.processing_stats['total_processing_time'] / 
                max(self.processing_stats['queries_handled'], 1)
            )
        }

# FastAPI Application
app = FastAPI(
    title="LLM-Powered Query-Retrieval System",
    description="Production-ready document analysis system for insurance, legal, and compliance domains",
    version="2.0.0"
)

# System initialization
system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system when the API starts"""
    global system
    
    # Get Gemini API key (we're now using Gemini instead of OpenAI)
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("💡 Set it with: export GEMINI_API_KEY='your-key-here'")
        print("🔗 Get your Gemini API key from: https://makersuite.google.com/app/apikey")
        raise RuntimeError("Gemini API key required")
    
    # Initialize the complete system
    try:
        system = CompleteQueryRetrievalSystem(gemini_api_key)
        print("🎉 API startup complete - system ready!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        raise

# Request/Response models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class SystemStatusResponse(BaseModel):
    system_ready: bool
    current_document: Optional[str]
    stats: Dict
    average_processing_time: float

# Authentication
EXPECTED_TOKEN = "1946e5edb566278a8419b7529c46cd12f704f8d440a584ebd07201ec32fcbfd0"

def verify_token(authorization: Optional[str] = Header(None)):
    """Verify the Bearer token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "LLM Query-Retrieval System v2.0",
        "status": "healthy" if system and system.is_ready else "initializing",
        "features": [
            "Real PDF processing",
            "Vector semantic search", 
            "Google Gemini intelligent reasoning",
            "Explainable answers"
        ]
    }

@app.get("/api/v1/status", response_model=SystemStatusResponse)
async def get_system_status(token: str = Depends(verify_token)):
    """Get detailed system status and statistics"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return system.get_system_status()

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_queries(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process document queries
    
    WHY THIS ENDPOINT DESIGN:
    - Matches the exact specification from the problem statement
    - Handles authentication as required
    - Provides proper error responses
    - Supports background processing for optimization
    """
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Validate request
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL required")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question required")
        
        if len(request.questions) > 20:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Too many questions (max 20)")
        
        # Process the request
        result = system.process_document_and_queries(
            request.documents, 
            request.questions
        )
        
        # Check if processing was successful
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return QueryResponse(answers=result['answers'])
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": system is not None,
        "system_ready": system.is_ready if system else False
    }
    
    if system:
        health_status.update({
            "documents_processed": system.processing_stats['documents_processed'],
            "queries_handled": system.processing_stats['queries_handled']
        })
    
    return health_status

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": ["/", "/api/v1/hackrx/run", "/api/v1/status"]}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please try again or contact support"}

# Development server
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Complete LLM Query-Retrieval System...")
    print("📋 Requirements:")
    print("   - GEMINI_API_KEY environment variable must be set")
    print("   - All required packages must be installed")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")