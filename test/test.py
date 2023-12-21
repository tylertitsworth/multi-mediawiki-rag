import argparse
import main
import pytest


def test_multiwiki():
    wiki = main.MultiWiki()
    assert wiki.embeddings_model == "sentence-transformers/all-mpnet-base-v2"
    assert wiki.model == "volo"
    assert wiki.prompt == "What is a Tako?"
    assert wiki.source == "./sources"
    assert wiki.wikis == {
        "dnd5e": "",
        "eberron": "",
        "forgottenrealms": "",
        "planescape": ""
    }
    
def test_multiwiki_set_args():
    wiki = main.MultiWiki()
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-embed', dest='embed', action='store_false')
    wiki.set_args(parser.parse_args([]))
    print(wiki.args)
    assert wiki.args.embed == True

@pytest.mark.embed
def test_create_vector_db():
    main.create_vector_db(
        "sentence-transformers/all-mpnet-base-v2",
        "./sources",
        {"forgottenrealms": ""}
    )

@pytest.mark.ollama
def test_create_chain():
    chain = main.create_chain(
        "sentence-transformers/all-mpnet-base-v2",
        "volo"
    )
    res = chain("What is a Tako?")
    assert res["answer"] != ""
    assert res["source_documents"] != []

