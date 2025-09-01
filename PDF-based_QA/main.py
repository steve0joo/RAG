import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import glob # 특정한 문자규칙을 주면 해당 규칙 내의 모든 특정 파일을 가져와서 목록을 만든다
import os

# API KEY load-information(정보로드)
load_dotenv()

# create cache directory
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# file-uploading folder
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF-based QA")

# Run it only once for the first time
if "messages" not in st.session_state:
    # create for saving conversation records
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    # if any file wasn't uploaded
    st.session_state["chain"] = None

# sidebar
with st.sidebar:
    # reset button
    clear_btn = st.button("Reset Conversation")
    # file uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    selected_prompts = "prompts/pdf-rag.yaml"

# print previous conversations
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# add new message
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# cache store the file (for the time-consuming work)
@st.cache_resource(show_spinner="Processing the uploaded file...")
def embed_file(file):
    # save the file to cache directory
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 1.Load Document 
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 2. Split Document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 3. Embedding
    embeddings = OpenAIEmbeddings()

    # 4. DB creation and storage (DB 생성 및 저장)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 5. Retriever
    retriever = vectorstore.as_retriever()
    return retriever


# create chain
def create_chain(retriever):
    # prompt = load_prompt(prompt_filepath, encoding="utf-8")
    # 6. Create Prompt
    prompt = PromptTemplate.from_template(
        """ You are an assistant for for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 

    #Context:
    {context}

    #Question:
    {question}

    #Answer:"""
    )

    # 7. LLM 
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    # 8. 체인(Chain) 생성
    # 질문을 입력하면 질문이 retriever로 들어가서 문서검색이 되고 context로 입력된다
    # question의 경우, RunnablePassthrough이라서 곧장 전달된다
    chain =(
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# when the file is uploaded
if uploaded_file:
    # after uploading file, create retriever (time-consuming)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever)
    st.session_state["chain"] = chain


    
# if reset btn clicked
if clear_btn:
    st.session_state["messages"] = []

# print previous conversations
print_messages()

# User's Input 
user_input = st.chat_input("Say something")

# empty container for warring message
warning_msg = st.empty()

# if user's input entered
if user_input:
    # create chain
    chain = st.session_state["chain"]

    if chain is not None:
        # print user's input
        st.chat_message("user").write(user_input)
        # stream call
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # create empty container and stream the token
            container = st.empty()
            ai_answer=""

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # store conversation
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # warning message
        warning_msg.error("Upload the file")