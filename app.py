import streamlit as st
import getpass
import os, shutil
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
import time
import json

if not os.environ.get("OPENAI_API_KEY"):
    password = st.text_input("Enter your OpenAI key here", type="password")
    button = st.button("Confirm", type="primary")
    while not button:
        time.sleep(1)
    os.environ["OPENAI_API_KEY"] = password
    st.rerun(scope="app")




embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
collection_name = "Chunk_store"
chroma_dir = "./chroma_data"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=chroma_dir
)


text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=400, chunk_overlap=40
)

def document_updater(document, vector_store):
    vector_store.reset_collection()
    texts = text_splitter.create_documents([document])
    uuids = [str(uuid4()) for _ in range(len(texts))]
    vector_store.add_documents(documents=texts, ids=uuids)


def answer_generator(prompt, vector_store, llm):
    results = vector_store.similarity_search(prompt, k=5)
    relevant_docs = [res.page_content for res in results]
    joint_docs = "\n".join(relevant_docs)
    prompt = f"""Answer the question: {prompt}\n
    use only the following as context:\n {joint_docs}"""
    return json.loads(llm.invoke(prompt).model_dump_json())["content"]


def pdf_parser(document):
    loader = PyPDFLoader(document)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return "\n".join([page.page_content for page in pages])

st.title("RAG App")
input_type = st.radio(
    "Choose input type",
    ["Text field", "PDF file"],
    captions=[
        "Write or copy paste the document text",
        "Upload a pdf document"
    ],
)



if input_type == "Text field":
    document = st.text_area(label = "document", placeholder = "Type your document here")
elif input_type == "PDF file":
    document = st.file_uploader("upload a PDF file", type="pdf")
    if document:
        with NamedTemporaryFile(dir='.', suffix='.pdf', delete=False) as f:
            f.write(document.getbuffer())
            document = pdf_parser(f.name)
        os.remove(f.name)

button1 = st.button("update document", type="primary")
prompt = st.text_area(label = "prompt", placeholder = "Enter your prompt here")
button2 = st.button("generate answer", type="primary")

if document and button1:
    document_updater(document, vector_store)
if prompt and button2:
    st.write(answer_generator(prompt, vector_store, llm))


