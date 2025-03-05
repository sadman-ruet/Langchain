import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Cache function for PDF loader
@st.cache_data
def load_pdf(file):
    temp_path = f"./temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader(temp_path)
    return loader.load()

# Cache function for text splitting
@st.cache_data
def split_text(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    return text_splitter.split_documents(_data)

# Cache function for embeddings and vector store
@st.cache_resource
def create_vectorstore(_docs, _embeddings, index_name):
    return PineconeVectorStore.from_documents(_docs, _embeddings, index_name=index_name)

# Cache function for the retriever
@st.cache_resource
def create_retriever(_vectorstore):
    return _vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 50})

# Cache function for LLM
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=768)

# Upload file
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write(f"✅ **File uploaded:** `{uploaded_file.name}`")

    # Progress bar
    progress_bar = st.progress(0)
    
    with st.spinner("🔍 Processing the document... Please wait."):
        # Load PDF (cached)
        data = load_pdf(uploaded_file)
        progress_bar.progress(30)  # Update progress

        # Split text into chunks (cached)
        docs = split_text(data)
        progress_bar.progress(60)  # Update progress

        # Create embeddings & vector store (cached)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        index_name = "harrypotter-qna"  # Change this to your Pinecone index
        vectorstore = create_vectorstore(docs, embeddings, index_name)
        retriever = create_retriever(vectorstore)
        progress_bar.progress(90)  # Update progress

        # Load LLM (cached)
        llm = load_llm()

        # System prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three to five sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
        progress_bar.progress(100)  # Completed
        time.sleep(0.5)  # Pause for better UX
        progress_bar.empty()  # Remove progress bar
    
    # Chat UI
    st.subheader("💬 Chat with the Book")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    # Input field for user question
    question = st.chat_input("Ask a question about the book 📖")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Show a spinner while processing
        with st.spinner("🤖 Thinking..."):
            response = rag_chain.invoke({"input": question})
            answer = response.get("answer", "I couldn't generate an answer.")

        # Save response and display
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
    
    st.button("Clear Chat History 🧹", on_click=lambda: st.session_state.messages.clear())
