import os, uuid, shutil
import fitz
import base64, re


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


def document_updater(document, vector_store_text, vector_store_image, text_splitter):
    vector_store_text.reset_collection()
    vector_store_image.reset_collection()
    folder_path = "images/"
    for filename in os.listdir(folder_path):
        filename = os.path.join(folder_path, filename)
        if os.path.isfile(filename) or os.path.islink(filename):
            os.unlink(filename)  # Delete file or symbolic link
        elif os.path.isdir(filename):
            shutil.rmtree(filename) 

    document_text, images = pdf_parser(document)

    for filename in os.listdir("./"):
            file_type = filename.split(".")[-1]
            if file_type == "pdf":
                if os.path.isfile(filename) or os.path.islink(filename):
                    os.unlink(filename)  # Delete file or symbolic link

    texts = text_splitter.create_documents([document_text])
    uuids = [str(uuid.uuid4()) for _ in range(len(texts))]
    vector_store_text.add_documents(documents=texts, ids=uuids)
    if images:
        uuids = [str(uuid.uuid4()) for _ in range(len(images))]
        vector_store_image.add_images(uris=images, ids=uuids) 


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
            #img_filename = os.path.join(output_dir, f"page{page_num+1}_img{img_index+1}.png")
            img_filename = f"{output_dir}page{page_num+1}_img{img_index+1}.png"
            images.append(img_filename)
            pix.save(img_filename)
            pix = None  # free memory
    return "\n".join(pages), images


def load_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
def parse_mixed(s: str):
    # split on tokens like img(123), keeping the delimiters
    parts = re.split(r"(image\(\d+\))", s)
    for part in parts:
        if not part:
            continue
        m = re.match(r"image\((\d+)\)", part)
        if m:
            yield {"type": "img", "id": int(m.group(1))}
        else:
            yield {"type": "text", "text": part}