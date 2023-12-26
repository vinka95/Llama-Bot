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
import os

load_dotenv()

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def create_docs(filename="papers.txt"):
    """Loads the processed text file, splits text into chunks for easier loading and creates a vectorstore using FAISS module
    """
    loader = TextLoader(filename)
    papers_text = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(papers_text)

    db = FAISS.from_documents(docs, openai_embeddings)
    return db


def ask_chatbot(question, papers_db, k=2):
    """Main ChatBot functionality, takes in a query and answers the query with the local context provided through SOTA papers on LLama2.

    Args:
        question (_type_): query text
        papers_db (_type_): Vectorstore object of papers text
        k (int, optional): No of Most Similar Docs to be considered, Defaults to 2.

    Returns:
        _type_: _description_
    """
    paper_docs = papers_db.similarity_search(question, k=k)
    docs_page_content = " ".join([d.page_content for d in paper_docs])

    # Chat model instance is created
    chat = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Chat prompt is designed using prompt templates
    template = ( "You are a helpful assistant that can answer questions on Llama 2, based on a set of papers from Arxiv." )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt] )
    
    # Create an LLMChain 
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    
    response = chain.run(text=question, docs=docs_page_content)
    return response


# Function Calls
papers_db = create_docs()


# Enter specific questions related that you have on the topic of Llama
# --------------------REPLACE YOUR QUESTION HERE----------------------------------#
question = "Name at least 5 domain-specific LLMs that have been created by fine-tuning Llama-2."

response = ask_chatbot(question, papers_db)

print(response)
