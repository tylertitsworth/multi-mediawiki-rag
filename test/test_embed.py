from collections import namedtuple
import yaml
from embed import parse_args, load_config, rename_duplicates, load_documents

Document = namedtuple("Document", ["page_content", "metadata"])


def test_parse_args():
    "Test parse_args()."
    config = load_config()
    config = parse_args(config, ["--test-embed"])
    assert config["mediawikis"] == ["dnd5e"]
    assert config["data_dir"] == "./test_data"
    assert config["question"] == "What is the Armor Class of a Beholder?"


def test_load_config():
    "Test load_config()."
    with open("config.yaml", "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    config = load_config()
    assert data == config


def test_rename_duplicates():
    "Test rename_duplicates()."
    documents = [
        Document(page_content="document 1", metadata={"source": "mydoc"}),
        Document(page_content="document 2", metadata={"source": "mydoc"}),
    ]
    renamed_documents = rename_duplicates(documents)
    assert documents[0].page_content == renamed_documents[0].page_content
    assert documents[0].page_content != renamed_documents[1].page_content
    assert documents[0].metadata["source"] == renamed_documents[0].metadata["source"]
    assert documents[0].metadata["source"] != renamed_documents[1].metadata["source"]


def test_load_documents():
    "Test load_documents()."
    config = load_config()
    config = parse_args(config, ["--test-embed"])
    documents = load_documents(config)
    [beholder_page] = [
        document
        for document in documents
        if document.metadata["source"] == "Beholder - dnd5e"
    ]
    assert "From Monster Manual, page 28." in beholder_page.page_content
    assert {"source": "Beholder - dnd5e"} == beholder_page.metadata
