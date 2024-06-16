import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
# To load web pages as documents
from langchain_community.document_loaders import WebBaseLoader
# To split documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Vector store for storing document embeddings
from langchain_community.vectorstores import Chroma
# For generating embeddings and interacting with OpenAI's LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv  # To load environment variables
# For creating chat prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# To create retrieval chains
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# To combine document chains
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from a .env file
load_dotenv()
API = os.getenv('API')



def get_vectorstore_from_url(url):
    # Load the webpage content as a document
    loader = WebBaseLoader(url , header_template={"User-Agent":"Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Googlebot/2.1; +http://www.google.com/bot.html) Safari/537.36"})
    document = loader.load()

    # Split the document into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vector store from the document chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(model="text-embedding-3-small",api_key=API))

    return vector_store


def get_context_retriever_chain(vector_store):
    # Initialize the language model with specified parameters
    llm = ChatOpenAI(model="gpt-3.5-turbo",  api_key=API)

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    # Define a prompt for generating search queries based on the conversation
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    # Create a history-aware retriever chain
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    # Initialize the language model with specified parameters
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=API)

    # Define a prompt for answering user questions based on context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # Create a chain for combining documents
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retrieval chain using the retriever chain and the document chain
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    # Get the retriever chain from the session state vector store
    retriever_chain = get_context_retriever_chain(
        st.session_state.vector_store)
    # Get the conversational RAG (Retrieval-Augmented Generation) chain
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Generate a response based on the user input and chat history
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']


# Configure the Streamlit app
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar for entering the website URL
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Display information if no URL is entered
if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # Handle user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        # Get the response for the user query
        response = get_response(user_query)
        # Update the chat history with the user query and the bot response
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)