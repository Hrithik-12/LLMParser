# 🤖 LLM-Powered Document Analysis System

A production-ready document analysis system that combines **PDF processing**, **vector search**, and **LLM reasoning** to answer questions about insurance policies, legal documents, and other text-based content.

## ✨ Features

- 📄 **Real PDF Processing** - Downloads and extracts text from PDF documents
- 🔍 **Vector Semantic Search** - Uses FAISS and sentence transformers for intelligent content retrieval
- 🧠 **LLM Integration** - Supports both Google Gemini and OpenAI for intelligent reasoning
- 🚀 **FastAPI Backend** - Production-ready REST API with authentication
- 📊 **Explainable Answers** - Provides reasoning, confidence scores, and supporting evidence
- 🔒 **Secure** - Token-based authentication and environment variable management
- 📈 **Scalable** - Efficient vector indexing and caching

## 🏗️ Architecture

```
User Query → PDF Processing → Vector Search → LLM Reasoning → Structured Answer
```

1. **PDF Processor** (`pdf_processor.py`) - Downloads and extracts text from PDFs
2. **Vector Search** (`vector_search.py`) - Creates embeddings and performs semantic search
3. **LLM Processor** (`llm_processor.py`) - Uses Gemini/GPT for intelligent reasoning
4. **Complete System** (`complete_system.py`) - FastAPI server orchestrating all components

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Gemini API key (recommended) or OpenAI API key

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd llm_parser
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Get API Keys:**
   - **Gemini API** (Recommended): https://makersuite.google.com/app/apikey
   - **OpenAI API** (Alternative): https://platform.openai.com/api-keys

6. **Start the server:**
```bash
python complete_system.py
```

The API will be available at: http://localhost:8000

## 📖 API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Process Document Queries
```http
POST /api/v1/hackrx/run
Authorization: Bearer 1946e5edb566278a8419b7529c46cd12f704f8d440a584ebd07201ec32fcbfd0
Content-Type: application/json

{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "Does this policy cover knee surgery?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

#### System Status
```http
GET /api/v1/status
Authorization: Bearer 1946e5edb566278a8419b7529c46cd12f704f8d440a584ebd07201ec32fcbfd0
```

## 🧪 Testing

Run the production test:
```bash
python test_production.py
```

This will:
- Test API endpoints
- Process a sample document
- Validate LLM responses
- Save results to `production_test_results.json`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes (primary) |
| `OPENAI_API_KEY` | OpenAI API key | Optional (backup) |

### System Configuration

- **Model**: `gemini-1.5-flash` (fast and capable)
- **Vector Model**: `all-MiniLM-L6-v2` (efficient embeddings)
- **Chunk Size**: 1000 words with 200 word overlap
- **Search Results**: Top 3 most relevant chunks per query

## 🏢 Use Cases

### Insurance Industry
- **Policy Analysis**: "Does this cover dental procedures?"
- **Claims Processing**: "What documentation is required?"
- **Compliance**: "Are pre-existing conditions covered?"

### Legal Documents
- **Contract Review**: "What are the termination clauses?"
- **Compliance**: "What are the liability limitations?"
- **Due Diligence**: "What are the key obligations?"

### General Documents
- **Research**: Extract key information from reports
- **Summarization**: Get concise answers from long documents
- **Analysis**: Understand complex technical documentation

## 🔄 System Workflow

1. **Document Input**: User provides PDF URL and questions
2. **PDF Processing**: System downloads and extracts text
3. **Chunking**: Text is split into manageable segments
4. **Vector Indexing**: Creates searchable embeddings using FAISS
5. **Query Processing**: For each question:
   - Performs semantic search to find relevant chunks
   - Sends context and question to Gemini LLM
   - Returns structured answer with reasoning

## 📊 Response Format

```json
{
  "answers": [
    "Knee surgery is covered under this policy with prior authorization required. Conditions: Must be performed by network providers; Pre-authorization required 48 hours in advance. (Note: This analysis has high confidence based on policy section 4.2)"
  ]
}
```

Each answer includes:
- **Direct Answer**: Clear response to the question
- **Conditions**: Any limitations or requirements
- **Confidence Indicator**: When confidence is low
- **Supporting Evidence**: Relevant document excerpts (in detailed mode)

## 🛠️ Development

### Project Structure
```
llm_parser/
├── complete_system.py      # Main FastAPI application
├── llm_processor.py        # LLM integration (Gemini/OpenAI)
├── pdf_processor.py        # PDF downloading and processing
├── vector_search.py        # FAISS vector search engine
├── test_production.py      # Production testing script
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md             # This file
```

### Adding New Features

1. **New Document Types**: Extend `pdf_processor.py`
2. **Different LLM Models**: Modify `llm_processor.py`
3. **Enhanced Search**: Update `vector_search.py`
4. **API Endpoints**: Add to `complete_system.py`

## 🚨 Security

- **Environment Variables**: Never commit `.env` files
- **API Keys**: Use environment variables only
- **Authentication**: Bearer token required for all API calls
- **Input Validation**: All inputs are validated and sanitized

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure `GEMINI_API_KEY` is set in `.env`
   - Verify key is valid at https://makersuite.google.com/

2. **PDF Processing Failures**:
   - Check if PDF URL is publicly accessible
   - Ensure PDF is not password-protected

3. **Memory Issues**:
   - Large PDFs may require more RAM
   - Consider implementing chunked processing

4. **Slow Response Times**:
   - First query is slower due to model loading
   - Subsequent queries are faster with cached embeddings

### Performance Optimization

- Use SSD storage for faster model loading
- Increase RAM for larger document processing
- Consider GPU acceleration for vector operations
- Implement Redis caching for production use

## 📈 Metrics and Monitoring

The system tracks:
- Documents processed
- Query response times
- LLM token usage
- Error rates and types

Access metrics via `/api/v1/status` endpoint.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- **Sentence Transformers**: For semantic embeddings
- **FAISS**: For efficient vector search
- **Google Gemini**: For LLM capabilities
- **FastAPI**: For the web framework

---

**Built with ❤️ for intelligent document analysis**
