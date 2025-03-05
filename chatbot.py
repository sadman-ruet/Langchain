import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
index_name = "harrypotter-qna"

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    st.write("File uploaded:", uploaded_file.name)
    loader = PyPDFLoader("Books/got.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3, max_tokens=500)\
    
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
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    question=st.chat_input("Ask a question about the book")
    if st.session_state.messages:
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

    if question := st.session_state.get("question"):
        response = rag_chain.invoke({"input": "{question}"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response["answer"])