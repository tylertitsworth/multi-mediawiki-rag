from pathlib import Path
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from embed import parse_args, load_config, load_documents
from app import import_db, create_chain

if not torch.cuda.is_available():
    torch.set_num_threads(torch.get_num_threads() * 2)


def test_import_db():
    "Test import_db()."
    config = load_config()
    config = parse_args(config, ["--test-embed"])
    documents = load_documents(config)
    if not Path(config["data_dir"]).is_dir():
        embeddings = HuggingFaceEmbeddings(
            model_name=config["embeddings_model"], cache_folder="./model"
        )
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=config["data_dir"],
        )
        vectordb.persist()
    vectordb = import_db(config)
    assert Path("test_data").is_dir()
    assert vectordb.embeddings.model_name == config["embeddings_model"]
    assert vectordb.embeddings.cache_folder == "./model"


def test_create_chain():
    "Test create_chain()"
    config = load_config()
    config = parse_args(config, ["--test-embed"])
    chain = create_chain(config)
    res = chain.invoke(config["question"])
    assert res != ""
    # assert res["answer"] != ""
    # assert res["context"] != []
