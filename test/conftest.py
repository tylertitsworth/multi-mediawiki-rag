import pytest

optional_markers = {
    "ollama": {
        "help": "Test ollama backend with Langchain",
        "marker-descr": "Enable langchain tests with ollana",
        "skip-reason": "Test only runs with the --{} option.",
    },
}


def pytest_addoption(parser):
    for marker, info in optional_markers.items():
        parser.addoption(
            f"--{marker}", action="store_true", default=False, help=info["help"]
        )


def pytest_configure(config):
    for marker, info in optional_markers.items():
        config.addinivalue_line(
            "markers", f"{marker}: {info['marker-descr']}"
        )


def pytest_collection_modifyitems(config, items):
    for marker, info in optional_markers.items():
        if not config.getoption(f"--{marker}"):
            skip_test = pytest.mark.skip(reason=info["skip-reason"].format(marker))
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip_test)
