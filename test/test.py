from main import MultiWiki, create_chain, create_vector_db

import argparse
import pytest


def test_multiwiki():
    wiki = MultiWiki()
    assert wiki.embeddings_model == "sentence-transformers/all-mpnet-base-v2"
    assert wiki.model == "volo"
    assert wiki.question == "What is a Tako?"
    assert wiki.source == "./sources"
    assert wiki.wikis == {
        "dnd4e": "",
        "dnd5e": "",
        "darksun": "",
        "dragonlance": "",
        "eberron": "",
        "exandria": "",
        "greyhawk": "",
        "forgottenrealms": "",
        "planescape": "",
        "ravenloft": "",
        "spelljammer": "",
    }


def test_multiwiki_set_args():
    wiki = MultiWiki()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", dest="embed", action="store_false")
    wiki.set_args(parser.parse_args([]))
    print(wiki.args)
    assert wiki.args.embed == True


@pytest.mark.embed
def test_create_vector_db():
    create_vector_db(
        "sentence-transformers/all-mpnet-base-v2",
        "./sources",
        {"dnd5e": ""}
    )


@pytest.mark.ollama
def test_create_chain():
    wiki = MultiWiki()
    chain, llm = create_chain(wiki.embeddings_model, wiki.model)
    res = chain(wiki.question)
    assert res["answer"] != ""
    assert res["source_documents"] != []
