# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from typing import List
# from langchain.schema import Document


# #Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)

#     documents=loader.load()

#     return documents



# def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
#     """
#     Given a list of Document objects, return a new list of Document objects
#     containing only 'source' in metadata and the original page_content.
#     """
#     minimal_docs: List[Document] = []
#     for doc in docs:
#         src = doc.metadata.get("source")
#         minimal_docs.append(
#             Document(
#                 page_content=doc.page_content,
#                 metadata={"source": src}
#             )
#         )
#     return minimal_docs



# #Split the Data into Text Chunks
# def text_split(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks



# #Download the Embeddings from HuggingFace 
# def download_hugging_face_embeddings():
#     embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
#     return embeddings

# --- OCR + deprecation-safe imports ---
# src/helper.py


# src/helper.py

from typing import List
from langchain_core.documents import Document  # <- no langchain.schema import

# OCR + layout-aware PDF parsing
from unstructured.partition.pdf import partition_pdf

# use the non-deprecated packages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Optional: set Tesseract path ONLY if `tesseract` is not on PATH.
# Make sure this points to the actual tesseract.exe, NOT the installer .exe
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ====== REPLACES your old load_pdf_file ======
def load_pdf_file(data_folder: str) -> List[Document]:
    import glob, os
    all_docs: List[Document] = []

    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    for pdf in pdf_files:
        elements = partition_pdf(
            pdf,
            strategy="hi_res",              # OCR + layout-aware (tables/images)
            extract_images_in_pdf=True,     # OCR images inside pages
            include_page_breaks=False,
        )
        for el in elements:
            text = getattr(el, "text", None)
            if text and text.strip():
                all_docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={"source": pdf}
                    )
                )
    return all_docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


def text_split(extracted_data: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    # 384-dim, works with your Pinecone index (dimension=384)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
