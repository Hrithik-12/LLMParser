# Fixed Gemini LLM Processor
# This fixes the missing method error and improves rate limit handling

import google.generativeai as genai
from typing import List, Dict, Optional
import json
import os
from dataclasses import dataclass
import time

@dataclass
class LLMResponse:
    """
    Structured response from LLM processing
    """
    answer: str
    confidence: float
    supporting_evidence: List[str]
    reasoning: str
    conditions: List[str]
    coverage_decision: str  # "covered", "not_covered", "conditional", "unclear"

class GeminiLLMProcessor:
    """
    Fixed Gemini LLM processor with proper method implementations
    
    WHY WE FIX THIS:
    - The original code was missing the format_response_for_api method
    - Need better rate limit handling for free tier
    - Improved error recovery
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini LLM processor"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Model configuration for faster responses
        self.model_name = "gemini-1.5-flash"  # Fast model
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,  # Reduced for faster generation
            "top_k": 20,   # Reduced for speed
            "max_output_tokens": 500,  # Reduced for faster responses
        }
        
        # Rate limiting for free tier
        self.requests_per_minute = 10  # Conservative limit for free tier
        self.request_timestamps = []
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        
        print(f"🤖 Gemini LLM Processor initialized with model: {self.model_name}")
    
    def _check_rate_limit(self):
        """
        Check and enforce rate limits for free tier
        
        WHY THIS IS IMPORTANT:
        - Free tier has strict limits (15 requests/minute)
        - Need to pace requests to avoid quota errors
        - Better user experience with controlled delays
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60
        ]
        
        # If we're at the limit, wait
        if len(self.request_timestamps) >= self.requests_per_minute:
            oldest_request = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_request) + 1
            
            if wait_time > 0:
                print(f"⏳ Rate limit protection: waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(current_time)
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for Gemini"""
        return """You are an expert insurance policy analyst with deep knowledge of insurance terms, conditions, and coverage rules. Your role is to analyze insurance policy documents and provide accurate, detailed answers to user queries.

IMPORTANT GUIDELINES:
1. Always base your answers on the provided policy document excerpts
2. Be precise about coverage conditions and limitations  
3. Clearly state when information is unclear or missing
4. Use simple, clear language while maintaining accuracy
5. Always provide your reasoning and supporting evidence

OUTPUT FORMAT:
Provide your response as a JSON object with these fields:
{
    "answer": "Direct answer to the user's question",
    "confidence": 0.85,
    "supporting_evidence": ["Quote 1 from document", "Quote 2 from document"],
    "reasoning": "Step-by-step explanation of how you reached this conclusion",
    "conditions": ["Condition 1", "Condition 2"],
    "coverage_decision": "covered|not_covered|conditional|unclear"
}

COVERAGE DECISIONS:
- "covered": Clearly covered by the policy
- "not_covered": Explicitly excluded or not mentioned
- "conditional": Covered but with specific conditions/limitations
- "unclear": Insufficient information to make a determination"""
    
    def process_query(self, query: str, relevant_chunks: List[Dict]) -> LLMResponse:
        """
        Process a user query using relevant document chunks
        """
        try:
            print(f"🧠 Processing query with Gemini: {query[:50]}...")
            
            # Check rate limits before making request
            self._check_rate_limit()
            
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Create the prompt
            prompt = self._create_user_prompt(query, context)
            
            # Call Gemini API with retries
            response = self._call_gemini_api(prompt)
            
            # Parse and validate response
            parsed_response = self._parse_llm_response(response)
            
            print(f"✅ Gemini processing complete (confidence: {parsed_response.confidence})")
            return parsed_response
            
        except Exception as e:
            print(f"❌ Gemini processing failed: {e}")
            return self._create_fallback_response(query, str(e))
    
    def _prepare_context(self, relevant_chunks: List[Dict]) -> str:
        """Prepare document context for the LLM - optimized for speed"""
        if not relevant_chunks:
            return "No relevant document sections found."
        
        # Take only top 2 chunks for faster processing
        top_chunks = relevant_chunks[:2]
        
        context_parts = []
        for i, chunk in enumerate(top_chunks):
            text = chunk.get('text', '').strip()
            if text:
                # Truncate very long chunks for speed
                if len(text) > 1000:
                    text = text[:1000] + "..."
                context_parts.append(f"Section {i+1}: {text}")
        
        return "\n\n".join(context_parts)
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create the user prompt combining query and context"""
        system_prompt = self.create_system_prompt()
        
        user_prompt = f"""{system_prompt}

POLICY DOCUMENT SECTIONS:
{context}

USER QUESTION:
{query}

Please analyze the policy document sections above and provide a comprehensive answer to the user's question. Follow the JSON output format specified above."""
        
        return user_prompt
    
    def _call_gemini_api(self, prompt: str) -> str:
        """
        Make the actual API call to Gemini with improved error handling
        """
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    print("📊 Gemini response generated successfully")
                    return response.text.strip()
                else:
                    raise Exception("Empty response from Gemini")
                
            except Exception as e:
                error_message = str(e)
                
                # Handle rate limit errors specifically
                if "429" in error_message or "quota" in error_message.lower():
                    if attempt < max_retries - 1:
                        # Extract retry delay from error if available
                        retry_delay = 60  # Default to 1 minute for quota errors
                        print(f"⏳ Rate limit hit, waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("Rate limit exceeded: Please wait before making more requests")
                
                # Handle other errors
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⏳ API error, retrying in {delay} seconds... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise Exception(f"Gemini API failed after {max_retries} attempts: {error_message}")
    
    def _parse_llm_response(self, response_text: str) -> LLMResponse:
        """Parse and validate the LLM's JSON response"""
        try:
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)
            
            # Validate and set defaults for required fields
            return LLMResponse(
                answer=str(parsed.get('answer', 'Unable to determine from available information.')),
                confidence=max(0.0, min(1.0, float(parsed.get('confidence', 0.5)))),
                supporting_evidence=list(parsed.get('supporting_evidence', [])),
                reasoning=str(parsed.get('reasoning', 'Analysis not available.')),
                conditions=list(parsed.get('conditions', [])),
                coverage_decision=parsed.get('coverage_decision', 'unclear') if parsed.get('coverage_decision') in ['covered', 'not_covered', 'conditional', 'unclear'] else 'unclear'
            )
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            return self._extract_fallback_answer(response_text)
        except Exception as e:
            print(f"⚠️ Response parsing failed: {e}")
            return self._create_fallback_response("", str(e))
    
    def _extract_fallback_answer(self, response_text: str) -> LLMResponse:
        """Extract a basic answer when JSON parsing fails"""
        lines = response_text.strip().split('\n')
        answer = lines[0] if lines else "Unable to process response."
        
        return LLMResponse(
            answer=answer,
            confidence=0.3,
            supporting_evidence=[],
            reasoning="Extracted from non-structured response.",
            conditions=[],
            coverage_decision='unclear'
        )
    
    def _create_fallback_response(self, query: str, error: str) -> LLMResponse:
        """Create a fallback response when LLM processing fails entirely"""
        return LLMResponse(
            answer=f"Unable to process the query due to a technical issue: {error}",
            confidence=0.0,
            supporting_evidence=[],
            reasoning=f"Gemini processing failed: {error}",
            conditions=[],
            coverage_decision='unclear'
        )
    
    def format_response_for_api(self, llm_response: LLMResponse) -> str:
        """
        Format the LLM response for the API output
        
        THIS WAS THE MISSING METHOD CAUSING THE ERROR!
        """
        answer_parts = [llm_response.answer]
        
        # Add conditions if present
        if llm_response.conditions:
            conditions_text = "; ".join(llm_response.conditions)
            answer_parts.append(f"Conditions: {conditions_text}")
        
        # Add confidence indicator for low-confidence responses
        if llm_response.confidence < 0.5:
            answer_parts.append("(Note: This analysis has low confidence due to limited information)")
        
        return " ".join(answer_parts)

class EnhancedQueryProcessor:
    """
    Enhanced query processor that combines vector search with Gemini LLM reasoning
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize with Gemini LLM capabilities"""
        self.llm_processor = GeminiLLMProcessor(gemini_api_key)
        print("🚀 Enhanced Query Processor initialized")
    
    def process_query_with_context(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """
        Process a query using both vector search results and LLM reasoning
        """
        print(f"🔍 Processing enhanced query: {query[:50]}...")
        
        try:
            # Process with LLM
            llm_response = self.llm_processor.process_query(query, relevant_chunks)
            
            # Format for API response
            formatted_answer = self.llm_processor.format_response_for_api(llm_response)
            
            # Return detailed information for internal use
            return {
                'formatted_answer': formatted_answer,
                'detailed_response': llm_response,
                'source_chunks': len(relevant_chunks),
                'processing_successful': True
            }
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            return {
                'formatted_answer': f"Unable to process query: {str(e)}",
                'detailed_response': None,
                'source_chunks': len(relevant_chunks),
                'processing_successful': False
            }
    
    def batch_process_queries(self, queries: List[str], search_function) -> List[str]:
        """Process multiple queries efficiently with better error handling"""
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"📝 Processing query {i}/{len(queries)}")
            
            try:
                # Get relevant chunks from vector search
                relevant_chunks = search_function(query)
                
                # Process with LLM
                result = self.process_query_with_context(query, relevant_chunks)
                results.append(result['formatted_answer'])
                
                # Add small delay between queries to help with rate limits
                if i < len(queries):  # Don't delay after the last query
                    time.sleep(1)
                
            except Exception as e:
                print(f"❌ Failed to process query {i}: {e}")
                results.append(f"Unable to process query due to error: {str(e)}")
        
        return results

# Test function
def test_gemini_processor():
    """Test the fixed Gemini processor"""
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable to test")
        print("💡 You can get an API key from: https://makersuite.google.com/app/apikey")
        return
    
    print("🧪 Testing Fixed Gemini Processor...")
    print("=" * 50)
    
    # Create sample relevant chunks
    sample_chunks = [
        {
            'text': 'This policy covers surgical procedures including orthopedic surgeries, knee replacement, and joint procedures. Prior authorization is required for all elective surgeries.',
            'similarity_score': 0.85
        }
    ]
    
    try:
        processor = GeminiLLMProcessor(api_key)
        
        # Test query
        query = "Does this policy cover knee surgery?"
        
        print(f"🔍 Test query: {query}")
        
        # Process query
        response = processor.process_query(query, sample_chunks)
        
        # Test the format_response_for_api method
        formatted = processor.format_response_for_api(response)
        print(f"\n✅ Formatted Response: {formatted}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_gemini_processor()