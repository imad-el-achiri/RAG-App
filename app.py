import streamlit as st
import os, shutil
from langchain_openai import ChatOpenAI
#from uuid import uuid4
import uuid
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import CharacterTextSplitter
import fitz
#from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
import time
import json
from custom_embedder import ClipEmbeddings as CE
import base64, re



if not os.environ.get("OPENAI_API_KEY"):
    password = st.text_input("Enter your OpenAI key here", type="password")
    button = st.button("Confirm", type="primary")
    while not button:
        time.sleep(1)
    os.environ["OPENAI_API_KEY"] = password
    st.rerun(scope="app")




#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#You can view the full list here: https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_retrieval_results.csv
Clip_embedder = CE(model="ViT-B-16-SigLIP-512", pretrained="webli")

chroma_dir = "./chroma_data"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)


#This is the vector store for text
vector_store_text = Chroma(
    collection_name="Chunk_store",
    embedding_function=Clip_embedder,
    persist_directory=chroma_dir,
)

#This is the vector store for images
vector_store_image = Chroma(
    collection_name="image_store",
    embedding_function=Clip_embedder,
    persist_directory=chroma_dir,
)


text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=400, chunk_overlap=40, separator=" "
)

def document_updater(document):
    vector_store_text.reset_collection()
    vector_store_image.reset_collection()
    folder_path = "images/"
    #Remove the images left from older documents so they do not get stored in the vector store of the current one
    for filename in os.listdir(folder_path):
        filename = os.path.join(folder_path, filename)
        #print(filename)
        if os.path.isfile(filename) or os.path.islink(filename):
            os.unlink(filename)  # Delete file or symbolic link
        elif os.path.isdir(filename):
            shutil.rmtree(filename) 

    document_text, images = pdf_parser(document)
    #Delete the pdf file once it is parsed, because it is not needed anymore
    for filename in os.listdir("./"):
            file_type = filename.split(".")[-1]
            if file_type == "pdf":
                if os.path.isfile(filename) or os.path.islink(filename):
                    os.unlink(filename)  # Delete file or symbolic link
    #Embed and store texts and images in their respective vector stores
    texts = text_splitter.create_documents([document_text])
    uuids = [str(uuid.uuid4()) for _ in range(len(texts))]
    vector_store_text.add_documents(documents=texts, ids=uuids)
    if images:
        uuids = [str(uuid.uuid4()) for _ in range(len(images))]
        vector_store_image.add_images(uris=images, ids=uuids)    

def load_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
# split on tokens like img(123), keeping the delimiters
def parse_mixed(s: str):
    parts = re.split(r"(image\(\d+\))", s)
    for part in parts:
        if not part:
            continue
        m = re.match(r"image\((\d+)\)", part)
        if m:
            yield {"type": "img", "id": int(m.group(1))}
        else:
            yield {"type": "text", "text": part}

def answer_generator(prompt, vector_store_text, vector_store_image, llm):
    text_results = vector_store_text.similarity_search(prompt, k=5)
    image_results = vector_store_image.similarity_search(prompt, k=3)
    relevant_docs = [res.page_content for res in text_results]
    relevant_images = [image.page_content for image in image_results]
    joint_docs = "\n\n\n".join(relevant_docs)
    system_prompt = """You are a helpful assistant. You will be provided with text context and potentially 
    one or more images. Each image will be identified by a specific code, like image(CODE_XYZ). 
    If you need to refer to any of these provided images in your response, 
    you MUST mention it in its own line as image(CODE_XYZ). The CODE_XYZ will be shared with before any image,
    So do not try to show images in a different way.
    Here is an example:
    The car engines are very complicated, as mentionned in the figure below:
    image(2)
    They are composed of many pieces, and then gets connected to a gearbox:
    image(7)
    which...

    Also, whenever you encounter a LaTex formula/code in the given context, and you decide to put it in your answer, make sure
    you put it inside double dollars, for example:
    x_b^2 becomes $$x_b^2$$
    $\sqrt{d_k}$ becomes $$\sqrt{d_k}$$
    $$frac{a}{b}$$ stays $$frac{a}{b}$$

    Focus on the content of the text and images to answer the user's query."""
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
    human_message_content.append({"type": "text", "text": "\nUser Query:\n" + prompt})

    human_message = HumanMessage(content=human_message_content)

    # Combine messages and invoke the model
    messages = [system_message, human_message]
    response = llm.invoke(messages)
    response = json.loads(response.model_dump_json())["content"]

    for token in parse_mixed(response):
        if token["type"] == "text":
            parts = re.split(r"(\$\$.+?\$\$)", token["text"])
            for part in parts:
                if part.startswith("$$") and part.endswith("$$"):
                    st.latex(part[2:-2])  # strip $$ and render
                else:
                    st.write(part)
            #st.write(token["text"])
        elif token["type"] == "img":
            img_id = token["id"]
            b64 = relevant_images[img_id]
            if b64 is None:
                st.error(f"⚠️ No image found for img({img_id})")
            else:
                st.image(base64.b64decode(b64), use_container_width=True)


def pdf_parser(document):
    doc = fitz.open(document)
    pages = []
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Extract text
        pages.append(page.get_text())
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            if pix.n > 4:  # convert CMYK to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            output_dir = f"images/{document.split("/")[-1].split(".")[0]}/"
            os.makedirs(output_dir, exist_ok=True)
            img_filename = f"{output_dir}page{page_num+1}_img{img_index+1}.png"
            images.append(img_filename)
            pix.save(img_filename)
            pix = None  # free memory
    return "\n".join(pages), images

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
prompt = st.text_area(label = "prompt", placeholder = "Enter your prompt here")
button2 = st.button("generate answer", type="primary")

if document and button1:
    #document_updater(document, vector_store, images)
    if document:
        with NamedTemporaryFile(dir='.', suffix='.pdf', delete=False) as f:
            f.write(document.getbuffer())
            document_updater(f.name)
if prompt and button2:
    st.write(answer_generator(prompt, vector_store_text, vector_store_image, llm))