import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Load the PDF files from the path
@st.cache_resource
def load_documents():
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

documents = load_documents()

# Split text into chunks
@st.cache_resource
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    return text_splitter.split_documents(documents)

text_chunks = split_documents(documents)

# Create embeddings
@st.cache_resource
def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 model_kwargs={'device':"cpu"})

embeddings = create_embeddings()

# Vector store
@st.cache_resource
def create_vector_store(text_chunks, embeddings):
    return FAISS.from_documents(text_chunks, embeddings)

vector_store = create_vector_store(text_chunks, embeddings)

# Create LLM
@st.cache_resource
def create_llm():
    return CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                         config={'max_new_tokens':128, 'temperature':0.01})

llm = create_llm()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)

st.title("Geo ChatBot")

def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about geology", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
