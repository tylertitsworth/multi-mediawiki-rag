from collections import Counter
from langchain.document_loaders import MWDumpLoader
from langchain.document_loaders.merge import MergedDataLoader
from main import MultiWiki, create_chain, create_vector_db, rename_duplicates

import argparse
import pytest
import shutil


def test_multiwiki():
    wiki = MultiWiki()
    assert wiki.data_dir == "./data"
    assert wiki.embeddings_model == "sentence-transformers/all-mpnet-base-v2"
    assert wiki.model == "volo"
    assert wiki.question == "How many eyestalks does a Beholder have?"
    assert wiki.source == "./sources"
    assert wiki.mediawikis == {
        # "dnd4e": "",
        "dnd5e": "",
        # "darksun": "",
        "dragonlance": "",
        "eberron": "",
        # "exandria": "",
        "greyhawk": "",
        "forgottenrealms": "",
        # "planescape": "",
        # "ravenloft": "",
        # "spelljammer": "",
    }


def test_multiwiki_set_args():
    wiki = MultiWiki()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", dest="embed", action="store_false")
    wiki.set_args(parser.parse_args([]))
    print(wiki.args)
    assert wiki.args.embed == True


@pytest.mark.embed
def test_rename_duplicates():
    wiki = MultiWiki()
    source = wiki.source
    mediawikis = wiki.mediawikis
    for wiki in mediawikis.keys():
        mediawikis[wiki] = MWDumpLoader(
            encoding="utf-8",
            file_path=f"{source}/{wiki}_pages_current.xml",
            namespaces=[0],
            skip_redirects=True,
            stop_on_error=False,
        )
    loader_all = MergedDataLoader(loaders=mediawikis.values())
    documents = loader_all.load()

    doc_counter = Counter([doc.metadata["source"] for doc in documents])
    duplicates = {source: count for source, count in doc_counter.items() if count > 1}

    if len(duplicates) > 1:
        documents_renamed = rename_duplicates(documents)
        doc_names = [getattr(item, "metadata")["source"] for item in documents_renamed]
        for dup in duplicates.items():
            for i in range(1, dup[1]):
                assert f"{dup[0]}_{i}" in doc_names
        assert len(documents) == len(documents_renamed)


@pytest.mark.embed
def test_create_vector_db():
    create_vector_db(
        "test_data", "sentence-transformers/all-mpnet-base-v2", "./sources", {"dnd5e": ""}
    )
    shutil.rmtree("test_data")


@pytest.mark.ollama
def test_create_chain():
    wiki = MultiWiki()
    chain, llm = create_chain(wiki.embeddings_model, wiki.model)
    res = chain(wiki.question)
    assert res["answer"] != ""
    assert res["source_documents"] != []
