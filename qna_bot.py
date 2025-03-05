import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
# Load API key securely (Replace with st.secrets or dotenv for production)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing API Key! Set GOOGLE_API_KEY in environment variables or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Define the Chat Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Initialize the LLM model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
output_parser = StrOutputParser()

# Streamlit UI
st.title("LangChain Demo with Gemini API")
input_text = st.text_input("Search the topic you want:")

if input_text:
    try:
        response = prompt | llm | output_parser
        output = response.invoke({"question": input_text})
        st.write(output)
    except Exception as e:
        st.error(f"Error: {e}")
