from main import MultiWiki, create_chain

import argparse
import pytest

import torch
torch.set_num_threads(22)

wiki = MultiWiki()


def test_multiwiki():
    assert wiki.data_dir == "./data"
    assert wiki.embeddings_model == "sentence-transformers/all-mpnet-base-v2"
    assert wiki.introduction == "Ah my good fellow!"
    assert wiki.model == "volo"
    assert wiki.question == "How many eyestalks does a Beholder have?"
    assert wiki.source == "./sources"
    assert wiki.mediawikis == [
        # "dnd4e",
        "dnd5e",
        # "darksun",
        "dragonlance",
        "eberron",
        # "exandria",
        "forgottenrealms",
        "greyhawk",
        # "planescape",
        # "ravenloft",
        # "spelljammer",
    ]


def test_multiwiki_set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", dest="embed", action="store_false")
    wiki.set_args(parser.parse_args([]))
    print(wiki.args)
    assert wiki.args.embed == True


@pytest.mark.ollama
def test_create_chain():
    chain = create_chain()
    res = chain(wiki.question)
    assert res["answer"] != ""
    assert res["source_documents"] != []
