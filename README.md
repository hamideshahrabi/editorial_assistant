# CBC Editorial Assistant

An AI-powered editorial assistant that helps journalists and editors follow CBC's editorial guidelines and access relevant information from articles and policies.

## Features

- **Policy Question Answering**: Get detailed answers about CBC's editorial guidelines, including:
  - Anonymous sources usage
  - Headline writing requirements
  - Social media guidelines
  - Food bank coverage policies

- **Article Information Retrieval**: Access relevant information from CBC articles with proper citations

- **Fast and Efficient**: Built with lightweight models for quick responses without GPU requirements

## Technical Details

- **Models Used**:
  - SentenceTransformer (all-MiniLM-L6-v2) for semantic search
  - Flan-T5-base for answer generation

- **Key Components**:
  - FastAPI backend
  - Vector store for efficient similarity search
  - Structured policy and article data handling

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/hamideshahrabi/editorial_assistant.git
cd editorial_assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the server:
```bash
python src/api/main.py
```

The server will start at `http://localhost:8000`

## API Usage

### Question Answering Endpoint

```bash
curl -X POST "http://localhost:8000/api/qa" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the guidelines for using anonymous sources?"}'
```

Example questions:
- "What are the guidelines for using anonymous sources?"
- "How should headlines be written?"
- "What are the social media guidelines?"
- "How should food bank coverage be handled?"

## Project Structure

```
editorial_assistant/
├── data/
│   ├── articles.json    # Article database
│   └── policies.txt     # Editorial guidelines
├── src/
│   ├── api/
│   │   └── main.py      # FastAPI server
│   └── generation/
│       └── text_generator.py
├── tests/               # Test files
└── requirements.txt     # Project dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.