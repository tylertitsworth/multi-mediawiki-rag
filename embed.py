import sys
from collections import namedtuple
import argparse
import yaml
import torch
from langchain_community.document_loaders import MWDumpLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


Document = namedtuple("Document", ["page_content", "metadata"])
if not torch.cuda.is_available():
    torch.set_num_threads(torch.get_num_threads() * 2)


def parse_args(config, args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-embed", dest="test_embed", action="store_true")
    args = parser.parse_args(args)
    if args.test_embed:
        config["mediawikis"] = ["dnd5e"]
        config["data_dir"] = "./test_data"
        config["question"] = "What is the Armor Class of a Beholder?"

    return config


def load_config():
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


def rename_duplicates(documents):
    document_counts = {}
    for idx, doc in enumerate(documents):
        doc_source = doc.metadata["source"]
        count = document_counts.get(doc_source, 0) + 1
        document_counts[doc_source] = count
        documents[idx].metadata["source"] = (
            doc_source if count == 1 else f"{doc_source}_{count - 1}"
        )

    return documents


def load_documents(config):
    documents = []
    for dump in config["mediawikis"]:
        # https://python.langchain.com/docs/integrations/document_loaders/mediawikidump
        loader = MWDumpLoader(
            encoding="utf-8",
            file_path=f'{config["source"]}/{dump}_pages_current.xml',
            # https://www.mediawiki.org/wiki/Help:Namespaces
            namespaces=[0],
            skip_redirects=True,
            stop_on_error=False,
        )
        # For each Document provided:
        # Modify the source metadata by accounting for duplicates (<name>_n)
        # And add the mediawiki title (<name>_n - <wikiname>)
        documents.extend(
            Document(
                doc.page_content, {"source": doc.metadata["source"] + f" - {dump}"}
            )
            for doc in rename_duplicates(loader.load())
        )

    return documents


if __name__ == "__main__":
    config = load_config()
    config = parse_args(config, sys.argv[1:])
    print("Loading Documents")
    documents = load_documents(config)
    print(f"Embedding {len(documents)} Documents, this may take a while.")
    # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embeddings_model"], cache_folder="./model"
    )
    # https://python.langchain.com/docs/integrations/vectorstores/chroma
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config["data_dir"],
    )
    vectordb.persist()
