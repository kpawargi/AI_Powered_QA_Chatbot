# 📄 AI-Powered PDF QA System using LangChain, Hugging Face, and AstraDB

This project is a command-line based AI-powered Question Answering system that allows users to upload a PDF and interact with it using natural language. It leverages LangChain, Hugging Face Transformers, and AstraDB to provide context-aware answers using Retrieval Augmented Generation (RAG).

## 🚀 Tech Stack
- Python  
- LangChain  
- Hugging Face Transformers  
- Retrieval Augmented Generation (RAG)  
- AstraDB  
- CassIO (for Cassandra/AstraDB integration)

## ✨ Key Features

- 📄 **Natural Language QA from PDFs**: Users can ask questions about PDF content using a conversational interface.
- 🧾 **Text Chunking**: Uses `PyPDF2` for PDF text extraction and LangChain’s `CharacterTextSplitter` for efficient chunking.
- 🔍 **Semantic Search with Embeddings**: Text chunks are embedded using `intfloat/e5-base-v2` and stored in AstraDB.
- 📦 **Vector Store Integration**: AstraDB is connected through `CassIO` and LangChain’s `Cassandra` vector store.
- 💬 **CLI Interface**: Enables real-time question answering from the command line.

## 📁 File Structure

```text
AI_Powered_QA_Chatbot/
├── app.py             # Main script for running the CLI-based QA system
└── requirements.txt   # List of Python dependencies
```

## 🛠️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/kpawargi/AI_Powered_QA_Chatbot.git
cd AI_Powered_QA_Chatbot
```

2. **(Optional) Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install the required dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Follow the prompts to upload a PDF and ask questions!**

## 🌐 Hugging Face Embedding Model Used

- [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)

## 📚 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/models)
- [AstraDB](https://www.datastax.com/astra)
- [CassIO](https://cassio.org/)
