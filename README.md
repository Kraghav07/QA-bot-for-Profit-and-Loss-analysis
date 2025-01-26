# QA-bot-for-Profit-and-Loss-analysis
# P&L 2024 Insights Assistant

Welcome to the **P&L 2024 Insights Assistant**, an AI-powered application built with Streamlit to extract and analyze information from PDF documents. This assistant enables users to upload a PDF file, ask questions, and retrieve contextually accurate answers based on the document content.

## Features

### 1. 📂 File Upload
- Upload a PDF document for analysis.
- Automatically processes and chunks the document into manageable sections for efficient querying.

### 2. ❓ Query
- Ask questions related to the uploaded PDF.
- Leverages Cohere embeddings and FAISS for semantic search.
- Returns precise answers by analyzing the most relevant sections of the document.

### 3. 📝 Query History
- Keeps a log of all user queries and answers during the session.
- Option to clear history.

## Tech Stack

### Frontend
- **Streamlit**: For building an intuitive and responsive user interface.

### Backend
- **Cohere**: For generating embeddings and handling natural language queries.
- **FAISS**: For efficient similarity search and indexing.
- **LangChain**: For chunking documents and managing the question-answering pipeline.
- **PyPDFLoader**: For extracting text from PDF files.

### Other Tools
- **dotenv**: For managing API keys securely.
- **Pandas**: For handling query history as a DataFrame.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Kraghav07/qa-insights-assistant.git
   cd qa-insights-assistant
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   - Create a `.env` file in the project root directory.
   - Add your Cohere API key:
     ```env
     COHERE_API_KEY=your_cohere_api_key
     ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the App**
   Open your browser and go to `http://localhost:8501`.

## Usage

1. Navigate to the **File Upload** tab to upload a PDF document.
2. Go to the **Query** tab and type your question.
3. View the answer and review query history in the **History** tab.

## Folder Structure
```
qa-insights-assistant/
├── app.py               # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not included in the repo)
├── README.md            # Documentation
└── data/                # (Optional) Folder for storing uploaded files
```

## Future Improvements
- **Multi-file Support**: Allow analysis of multiple files simultaneously.
- **Enhanced Visualizations**: Provide graphs or tables for financial data insights.
- **Authentication**: Add user authentication for a personalized experience.

