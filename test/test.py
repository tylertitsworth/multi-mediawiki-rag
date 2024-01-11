import argparse
import pytest
import torch
from main import MultiWiki, create_chain, set_chat_settings

if not torch.cuda.is_available():
    torch.set_num_threads(torch.get_num_threads() * 2)

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
    parser.add_argument("--test-embed", dest="test_embed", action="store_true")
    wiki.set_args(parser.parse_args([]))
    print(wiki.args)
    assert wiki.args.embed is True


def test_set_chat_settings():
    global wiki
    init_settings = {
        setting: wiki.__dict__[setting]
        for setting in ["repeat_penalty", "temperature", "top_k", "top_p"]
    }
    settings = {
        setting: wiki.__dict__[setting] + 1
        for setting in ["repeat_penalty", "temperature", "top_k", "top_p"]
    }
    wiki = set_chat_settings(settings)
    assert wiki.repeat_penalty == init_settings["repeat_penalty"] + 1
    assert wiki.temperature == init_settings["temperature"] + 1
    assert wiki.top_k == init_settings["top_k"] + 1
    assert wiki.top_p == init_settings["top_p"] + 1


@pytest.mark.ollama
def test_create_chain():
    chain = create_chain()
    res = chain(wiki.question)
    assert res["answer"] != ""
    assert res["source_documents"] != []
