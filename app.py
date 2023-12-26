from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import streamlit as st
import os


load_dotenv()

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def create_docs(filename="papers.txt"):
    loader = TextLoader(filename)
    papers_text = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(papers_text)

    db = FAISS.from_documents(docs, openai_embeddings)
    return db


def ask_chatbot(question, papers_db, k=2):
    
    paper_docs = papers_db.similarity_search(question, k=k)
    docs_page_content = " ".join([d.page_content for d in paper_docs])
    
    chat = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    template = ( "You are a helpful assistant that can answer questions on Llama 2, based on a set of papers from Arxiv." )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt] )
    # chat_prompt = chat_prompt.format_prompt(text=question).to_messages()
    
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    # response = chat(chat_prompt.format_prompt(text=question).to_messages())
    
    response = chain.run(text=question, docs=docs_page_content)
    
    print(response)

    return response

st.title("Ask me anything on Llama 2")

papers_db = create_docs()

form = st.form("chat_form")
form.write("Inside the form")

# question = st.text_input("You can write your questions and hit Enter to know more about Llama2!", key="input")    

   # Every form must have a submit button.
# submitted = st.form_submit_button("Submit")
   
# if submitted:
# response = ask_chatbot(question, papers_db)
    #    response = response.replace("\n", "")
    #    st.text_area("Response:", value=response)

st.write("Outside the form")