import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
import cassio
import os
import uuid
from dotenv import load_dotenv

# 🔐 Load secrets from environment (Hugging Face Spaces uses HF Secrets)
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 🧠 Initialize AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="Query PDF with LangChain", layout="wide")
st.title("📄💬 Query PDF using LangChain + AstraDB (Hugging Face Models)")

# 📁 PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully.")
    process_button = st.button("🔄 Process PDF")

    if process_button:
        # 🧾 Read PDF
        pdf_reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        # ✂️ Split into Chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=800, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_text(raw_text)

        # 🧠 Embeddings
        embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

        # 🤖 LLM
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
        )

        # 🗃️ Unique Table Name for Each PDF Upload
        table_name = "qa_" + str(uuid.uuid4()).replace("-", "_")

        # 📦 Vector Store Setup
        vector_store = Cassandra(
            embedding=embedding,
            table_name=table_name,
            session=None,
            keyspace=None,
        )

        vector_store.add_texts(texts[:50])
        st.success(f"📚 {len(texts[:50])} chunks embedded and stored in AstraDB.")

        # 🔍 Setup Index
        astra_vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

        # 💬 Ask Questions
        st.header("🤖 Ask a question about your PDF")
        user_question = st.text_input("💬 Type your question here")

        if user_question:
            with st.spinner("🧠 Thinking..."):
                try:
                    # Retrieve relevant context (used internally, not displayed)
                    retrieved_docs = vector_store.similarity_search(user_question, k=8)
                    if not retrieved_docs:
                        st.warning("⚠️ No relevant text found. Try rephrasing your question.")
                    else:
                        answer = astra_vector_index.query(user_question, llm=llm)
                        if answer.strip():
                            st.markdown("### 🧠 Answer:")
                            st.write(answer.strip())
                        else:
                            st.warning("⚠️ Model returned an empty response.")
                except Exception as e:
                    st.error(f"🚨 Error: {str(e)}")
