import streamlit as st
import openai
from config import API_KEY
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from chatTemplate import css, bot_template, user_template
from datetime import datetime
openai.api_key = API_KEY

def format_timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY) 
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=API_KEY) 

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            template = user_template
        else:
            template = bot_template

        timestamp = format_timestamp()
        if message.content:
            st.write(template.replace("{{TIMESTAMP}}", timestamp).replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # If bot couldn't find an answer
            fallback_message = "I'm sorry, but I couldn't find a suitable response for your question."
            st.write(template.replace("{{TIMESTAMP}}", timestamp).replace("{{MSG}}", fallback_message), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="SmartStudy Assistant", page_icon=":book:")
    st.write(css,unsafe_allow_html=True)
    session_state = st.session_state
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Smart Study Assistant")
    st.markdown("Welcome to SmartStudy Assistant! Ask questions about your study materials and get personalized answers. Upload your PDFs to begin the conversation.")

    user_question= st.text_input("Ask questions about your documents")

    if user_question: 
        handle_userinput(user_question) 
 

    with st.sidebar:
        st.subheader("Your Study Materials")
        pdf_docs =  st.file_uploader("Upload your PDF here", accept_multiple_files=True)
        if st.button("Process"):
           with st.spinner("Processing"):
                # get pdf text
                raw_text=get_text_from_pdf(pdf_docs)
                

                # get the text chunks
                text_chunks = get_text_chunks(raw_text) 

                # get vector store
                vectorstore = get_vectorstore(text_chunks)

                # Conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()