import sys
from pathlib import Path
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from embed import load_config, rename_duplicates

mydocs = [
    "path/to/document1.txt",
    "path/to/another/document2.md",
    "document3.py",
]


if __name__ == "__main__":
    config = load_config()
    config["add_docs"] = mydocs
    documents = []
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embeddings_model"], cache_folder="./model"
    )
    if Path(config["data_dir"]).is_dir():
        print(f'Database at {config["data_dir"]} not found')
        sys.exit(1)
    vectordb = Chroma(
        persist_directory=config["data_dir"], embedding_function=embeddings
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=False
    )
    for doc in config["add_docs"]:
        if Path(doc).is_file():
            if doc.endswith(".md"):
                # pip install unstructured
                # pip install markdown
                # https://python.langchain.com/docs/modules/data_connection/document_loaders/markdown
                # https://python.langchain.com/docs/modules/data_connection/document_transformers/markdown_header_metadata
                split_headers = md_splitter.split_text(doc)
                split_docs = text_splitter.create(documents[split_headers])
                documents.extend(
                    UnstructuredMarkdownLoader(rename_duplicates(split_docs)).load()
                )
            elif doc.endswith(".pdf"):
                # pip install pypdf
                # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
                # https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
                split_docs = text_splitter.create_documents([doc])
                documents.extend(PyPDFLoader(rename_duplicates(split_docs)).load())
            else:
                # https://python.langchain.com/docs/modules/data_connection/document_loaders/
                # https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
                split_docs = text_splitter.create_documents([doc])
                documents.extend(TextLoader(rename_duplicates(split_docs)).load())
        else:
            print(f"Input Document: {doc} not found")
            sys.exit(1)
    Chroma.add_documents(
        vectordb,
        documents=documents,
    )
    vectordb.persist()
    print(f'{config["add_docs"]} successfully added to {config["data_dir"]}')
