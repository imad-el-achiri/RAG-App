import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from tempfile import NamedTemporaryFile
import time
import json
from custom_embedder import ClipEmbeddings as CE
import base64, re
from utility_functions import document_updater, parse_mixed, system_prompt


if not os.environ.get("OPENAI_API_KEY"):
    password = st.text_input("Enter your OpenAI key here", type="password")
    button = st.button("Confirm", type="primary")
    while not button:
        time.sleep(1)
    os.environ["OPENAI_API_KEY"] = password
    st.rerun(scope="app")


#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
@st.cache_resource
def Clip_embedder_creator(model_name:str, pretrained:str) -> CE:
    return CE(model_name, pretrained)

@st.cache_resource
def vector_store_creator(text_collection_name:str="Chunk_store",
                         image_collection_name:str="image_store",
                         persist_directory:str = "./chroma_data"):
    return [Chroma(
        collection_name=text_collection_name,
        embedding_function=Clip_embedder,
        persist_directory=persist_directory,
    ),
    Chroma(
            collection_name=image_collection_name,
            embedding_function=Clip_embedder,
            persist_directory=persist_directory,
        )]

@st.cache_resource
def text_splitter_creator(encoding_name:str="cl100k_base", chunk_size:int=400,
                          chunk_overlap:int=40, separator:str=" "):
    return CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator
    )
@st.cache_resource
def OpenAI_llm_connection_creator(model:str="gpt-4o-mini", temperature:float=0.6):
    return ChatOpenAI(model=model, temperature=temperature)


Clip_embedder = Clip_embedder_creator(model_name="ViT-B-16-SigLIP-512", pretrained="webli")

vector_store_text, vector_store_image = vector_store_creator()

llm = OpenAI_llm_connection_creator()

text_splitter = text_splitter_creator()


def answer_display(model_response:str, relevant_images:list):
    for token in parse_mixed(model_response):
        if token["type"] == "text":
            parts = re.split(r"(\$\$.+?\$\$)", token["text"])
            for part in parts:
                if part.startswith("$$") and part.endswith("$$"):
                    st.latex(part[2:-2])  # strip $$ and render
                else:
                    st.write(part)
        elif token["type"] == "img":
            img_id = token["id"]
            b64 = relevant_images[img_id]
            if b64 is None:
                st.error(f"⚠️ No image found for img({img_id})")
            else:
                st.image(base64.b64decode(b64), use_container_width=True)


def prompt_generator(user_prompt:str, system_prompt:str,vector_store_text, vector_store_image)->str:
    text_results = vector_store_text.similarity_search(user_prompt, k=5)
    image_results = vector_store_image.similarity_search(user_prompt, k=3)

    relevant_docs = [res.page_content.strip() for res in text_results]
    relevant_images = [image.page_content for image in image_results]

    joint_docs = "\n\n\n".join(relevant_docs)

    system_message = SystemMessage(content=system_prompt)

    # Construct the content for the HumanMessage
    human_message_content = []

    # 1. Add the initial text context
    human_message_content.append({"type": "text", "text": joint_docs})
    human_message_content.append({"type": "text", "text": "\n\n--- End of initial text context --- \n\nImage contexts follow:"})

    # 2. Add images with their metadata codes
    if not relevant_images:
         human_message_content.append({"type": "text", "text": "No images were provided in this context."})
    else:
        for idx, image in enumerate(relevant_images):            
                human_message_content.append({
                    "type": "text",
                    "text": f"The following image is identified as image({idx}):"
                })
                human_message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"}
                })
        human_message_content.append({"type": "text", "text": "\n\n--- End of image contexts --- \n"})

    # 3. Add the user's query
    human_message_content.append({"type": "text", "text": "\nUser Query:\n" + user_prompt})

    human_message = HumanMessage(content=human_message_content)

    # Combine messages and invoke the model
    final_prompt = [system_message, human_message]
    return final_prompt, relevant_images



def answer_generator(user_prompt, vector_store_text, vector_store_image, llm):
    final_prompt, relevant_images = prompt_generator(user_prompt, system_prompt,vector_store_text, vector_store_image)
    response = llm.invoke(final_prompt)
    response = json.loads(response.model_dump_json())["content"]
    return response, relevant_images

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
    images=None
elif input_type == "PDF file":
    document = st.file_uploader("upload a PDF file", type="pdf")


button1 = st.button("update document", type="primary")
user_prompt = st.text_area(label = "prompt", placeholder = "Enter your prompt here")
button2 = st.button("generate answer", type="primary")

if document and button1:    
    if document:
        with NamedTemporaryFile(dir='.', suffix='.pdf', delete=False) as f:
            f.write(document.getbuffer())            
            document_updater(f.name, vector_store_text, vector_store_image, text_splitter)
if user_prompt and button2:
    answer, relevant_images = answer_generator(user_prompt, vector_store_text, vector_store_image, llm)
    answer_display(answer, relevant_images)