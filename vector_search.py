# Real Vector Search with FAISS
# This replaces keyword matching with semantic search using embeddings

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
import os

class VectorSearchEngine:
    """
    Real vector search engine using FAISS and sentence transformers
    
    WHY WE USE VECTOR SEARCH:
    - Semantic understanding: "knee surgery" matches "orthopedic procedure"
    - Better relevance: Finds contextually similar content
    - Scalable: FAISS handles millions of vectors efficiently
    - Industry standard: Used by major companies for document search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector search engine
        
        WHY THIS MODEL:
        - all-MiniLM-L6-v2 is fast and efficient
        - Good balance of speed vs accuracy
        - Works well for insurance/legal documents
        - Small enough to run on modest hardware
        """
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        print(f"🤖 Initializing embedding model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """
        Load the sentence transformer model
        
        WHY WE LOAD MODELS SEPARATELY:
        - Models are large and take time to load
        - We want to handle loading errors gracefully
        - Allows for model swapping without recreating the class
        """
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def create_embeddings(self, text_chunks: List[Dict]) -> np.ndarray:
        """
        Create embeddings for all text chunks
        
        WHY WE BATCH PROCESS:
        - More efficient than processing one by one
        - Better GPU utilization if available
        - Consistent embedding quality across chunks
        """
        if not text_chunks:
            return np.array([])
        
        print(f"🔢 Creating embeddings for {len(text_chunks)} chunks...")
        
        # Extract just the text content
        texts = [chunk['text'] for chunk in text_chunks]
        
        try:
            # Create embeddings in batches for efficiency
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=32,  # Process 32 chunks at a time
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"✅ Created embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Failed to create embeddings: {e}")
            raise
    
    def build_index(self, text_chunks: List[Dict]) -> bool:
        """
        Build FAISS index from text chunks
        
        WHY WE USE FAISS:
        - Extremely fast similarity search
        - Handles large datasets efficiently
        - Used by Facebook, Google, and other tech giants
        - Supports different index types for different use cases
        """
        try:
            # Store chunks for later retrieval
            self.chunks = text_chunks.copy()
            
            # Create embeddings
            embeddings = self.create_embeddings(text_chunks)
            
            if embeddings.size == 0:
                print("❌ No embeddings created")
                return False
            
            # Create FAISS index
            print("🏗️  Building FAISS index...")
            
            # IndexFlatIP = Inner Product (cosine similarity for normalized vectors)
            # WHY THIS INDEX TYPE:
            # - Simple and accurate for small to medium datasets
            # - No training required
            # - Good baseline performance
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
            
            print(f"✅ FAISS index built with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"❌ Failed to build index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using vector similarity
        
        WHY VECTOR SEARCH WORKS BETTER:
        - Query: "Does this cover knee surgery?"
        - Matches: "orthopedic procedures", "surgical interventions"
        - Keyword search would miss these semantic connections
        """
        if not self.index or not self.chunks:
            print("❌ Index not built yet")
            return []
        
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Create embedding for the query
            query_embedding = self.embedding_model.encode([query])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.chunks))
            )
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0:  # Valid index
                    chunk = self.chunks[idx].copy()
                    chunk.update({
                        'similarity_score': float(score),
                        'rank': i + 1,
                        'query': query
                    })
                    results.append(chunk)
            
            print(f"✅ Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_search_summary(self, results: List[Dict]) -> Dict:
        """
        Create a summary of search results for debugging
        
        WHY WE NEED SUMMARIES:
        - Helps understand what the system found
        - Useful for debugging search quality
        - Provides confidence metrics
        """
        if not results:
            return {'total_results': 0, 'avg_score': 0.0}
        
        scores = [r['similarity_score'] for r in results]
        
        return {
            'total_results': len(results),
            'avg_score': np.mean(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'score_range': max(scores) - min(scores)
        }
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the FAISS index and chunks to disk
        
        WHY WE SAVE INDEXES:
        - Building indexes is expensive
        - Allows for persistence across restarts
        - Enables offline processing
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save chunks and metadata
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'model_name': self.model_name,
                    'dimension': self.dimension
                }, f)
            
            print(f"✅ Index saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load a previously saved FAISS index
        
        WHY WE LOAD INDEXES:
        - Skip expensive re-processing
        - Enable quick startup
        - Consistency across sessions
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load chunks and metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                # Verify model compatibility
                if data['model_name'] != self.model_name:
                    print(f"⚠️  Warning: Saved model ({data['model_name']}) differs from current ({self.model_name})")
            
            print(f"✅ Index loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load index: {e}")
            return False

class DocumentSearchSystem:
    """
    Complete document search system combining PDF processing and vector search
    
    WHY WE COMBINE THESE:
    - End-to-end document processing pipeline
    - Handles the complete workflow from URL to search results
    - Provides a clean interface for the API
    """
    
    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.is_ready = False
    
    def process_document(self, document_url: str, pdf_processor) -> bool:
        """
        Process a document and build search index
        
        WHY THIS WORKFLOW:
        1. Download and parse document
        2. Split into chunks
        3. Create embeddings
        4. Build searchable index
        """
        print(f"🚀 Processing document: {document_url}")
        
        # Extract text from document
        doc_result = pdf_processor.extract_text_from_url(document_url)
        
        if not doc_result['success']:
            print(f"❌ Document processing failed: {doc_result['error']}")
            return False
        
        # Create chunks
        chunks = pdf_processor.create_document_chunks(doc_result['text'])
        
        if not chunks:
            print("❌ No chunks created from document")
            return False
        
        # Build vector index
        success = self.vector_engine.build_index(chunks)
        
        if success:
            self.is_ready = True
            print("✅ Document processing complete - ready for queries!")
        
        return success
    
    def search_document(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search the processed document
        """
        if not self.is_ready:
            print("❌ System not ready - process a document first")
            return []
        
        return self.vector_engine.search(query, top_k)

# Example usage and testing
def test_vector_search():
    """Test the vector search system"""
    
    print("🧪 Testing Vector Search System...")
    print("=" * 50)
    
    # Create sample chunks (in real usage, these come from PDF processing)
    sample_chunks = [
        {
            'text': 'This policy covers all surgical procedures including orthopedic surgeries, knee replacements, and joint procedures. Prior authorization is required for elective surgeries.',
            'chunk_id': 0
        },
        {
            'text': 'Maternity benefits include coverage for normal delivery, cesarean section, and prenatal care. A waiting period of 9 months applies to maternity benefits.',
            'chunk_id': 1
        },
        {
            'text': 'Pre-existing diseases are covered after a waiting period of 36 months of continuous coverage. This includes diabetes, hypertension, and heart conditions.',
            'chunk_id': 2
        },
        {
            'text': 'The policy excludes cosmetic surgeries, experimental treatments, and treatments received outside the network hospitals.',
            'chunk_id': 3
        }
    ]
    
    # Initialize search system
    search_system = VectorSearchEngine()
    
    # Build index
    print("\n🏗️  Building index...")
    success = search_system.build_index(sample_chunks)
    
    if not success:
        print("❌ Failed to build index")
        return
    
    # Test queries
    test_queries = [
        "Does this policy cover knee surgery?",
        "What is the waiting period for pregnancy?",
        "Are pre-existing conditions covered?",
        "What treatments are not covered?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = search_system.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['similarity_score']:.3f}")
            print(f"     Text: {result['text'][:100]}...")

if __name__ == "__main__":
    test_vector_search()