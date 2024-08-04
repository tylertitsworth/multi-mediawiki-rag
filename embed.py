import argparse
import sys
from collections import namedtuple
from typing import Any

import torch
import yaml
from langchain_community.document_loaders import MWDumpLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.contrib.concurrent import process_map

Document = namedtuple("Document", ["page_content", "metadata"])
if not torch.cuda.is_available():
    torch.set_num_threads(torch.get_num_threads() * 2)


def parse_args(config: dict, args: list):
    """Parses command line arguments.

    Args:
        config (dict): items in config.yaml
        args (list(str)): user input parameters

    Returns:
        dict: dictionary of items in config.yaml, modified by user input parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-embed", dest="test_embed", action="store_true")
    args = parser.parse_args(args)
    if args.test_embed:
        config["mediawikis"] = ["dnd5e"]
        config["data_dir"] = "./test_data"
        config["question"] = "What is the Armor Class of a Beholder?"

    return config


def load_config():
    """Loads configuration from config.yaml file.

    Returns:
        dict: items in config.yaml
    """
    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: File config.yaml not found.")
        sys.exit(1)
    except yaml.YAMLError as err:
        print(f"Error reading YAML file: {err}")
        sys.exit(1)

    return data


def rename_duplicates(documents: [Document]):
    """Rename duplicates in a list of documents.

    Args:
        documents (list(Document)): input documents via loader.load()

    Returns:
        list(Document): input documents with modified source metadata
    """
    document_counts = {}
    for idx, doc in enumerate(documents):
        doc_source = doc.metadata["source"]
        count = document_counts.get(doc_source, 0) + 1
        document_counts[doc_source] = count
        documents[idx].metadata["source"] = (
            doc_source if count == 1 else f"{doc_source}_{count - 1}"
        )

    return documents


def load_document(wiki: tuple):
    """Loads an xml file of mediawiki pages into document format.

    Args:
        wiki (str): name of the wiki

    Returns:
        list(Document): input documents from mediawikis config with modified source metadata
    """
    # https://python.langchain.com/docs/integrations/document_loaders/mediawikidump
    loader = MWDumpLoader(
        encoding="utf-8",
        file_path=f"{wiki[0]}/{wiki[1]}_pages_current.xml",
        # https://www.mediawiki.org/wiki/Help:Namespaces
        namespaces=[0],
        skip_redirects=True,
        stop_on_error=False,
    )
    # For each Document provided:
    # Modify the source metadata by accounting for duplicates (<name>_n)
    # And add the mediawiki title (<name>_n - <wikiname>)

    return [
        Document(doc.page_content, {"source": doc.metadata["source"] + f" - {wiki[1]}"})
        for doc in rename_duplicates(loader.load())
    ]


class CustomTextSplitter(RecursiveCharacterTextSplitter):
    """Creates a custom Character Text Splitter.

    Args:
        RecursiveCharacterTextSplitter (RecursiveCharacterTextSplitter): Generates chunks based on different separator rules
    """

    def __init__(self, **kwargs: Any) -> None:
        separators = [r"\w(=){3}\n", r"\w(=){2}\n", r"\n\n", r"\n", r"\s"]
        super().__init__(separators=separators, keep_separator=False, **kwargs)


def load_documents(config: dict):
    """Load all the documents in the MediaWiki wiki page using multithreading.

    Args:
        config (dict): items in config.yaml

    Returns:
        list(Document): input documents from mediawikis config with modified source metadata
    """

    documents = sum(
        process_map(
            load_document,
            [(config["source"], wiki) for wiki in config["mediawikis"]],
            desc="Loading Documents",
            max_workers=torch.get_num_threads(),
        ),
        [],
    )
    splitter = CustomTextSplitter(
        add_start_index=True,
        chunk_size=1000,
        is_separator_regex=True,
    )
    documents = sum(
        process_map(
            splitter.split_documents,
            [[doc] for doc in documents],
            chunksize=1,
            desc="Splitting Documents",
            max_workers=torch.get_num_threads(),
        ),
        [],
    )
    documents = rename_duplicates(documents)

    return documents


if __name__ == "__main__":
    config = load_config()
    config = parse_args(config, sys.argv[1:])
    documents = load_documents(config)
    print(f"Embedding {len(documents)} Documents, this may take a while.")
    # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    embeddings = HuggingFaceEmbeddings(
        cache_folder="./model",
        model_name=config["embeddings_model"],
        show_progress=True,
    )
    # https://python.langchain.com/docs/integrations/vectorstores/chroma
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config["data_dir"],
    )
