import os

import chainlit as cl
from chainlit.input_widget import Slider, TextInput
from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
from langchain.globals import set_llm_cache
from langchain.schema.runnable.config import RunnableConfig
from langchain_chroma import Chroma
from langchain_community.cache import InMemoryCache
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from embed import load_config, parse_args


def import_db(config: dict):
    """Use existing Chroma vectorDB

    Args:
        config (dict): items in config.yaml

    Returns:
        Chroma: initialize a Chroma client.
    """
    # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    embeddings = HuggingFaceEmbeddings(
        cache_folder="./model",
        model_name=config["embeddings_model"],
    )
    vectordb = Chroma(
        persist_directory=config["data_dir"], embedding_function=embeddings
    )

    return vectordb


def create_chain(config: dict):
    """Creates a conversation chain from a config file.

    Args:
        config (dict): items in config.yaml

    Returns:
        Runnable: Langchain Runnable for use with ChatOllama
    """
    if os.getenv("TEST"):
        config = parse_args(config, ["--test-embed"])
        print("Running in TEST mode.")
    set_llm_cache(InMemoryCache())
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    vectordb = import_db(config)
    prompt = hub.pull("rlm/rag-prompt")
    # https://python.langchain.com/docs/integrations/llms/ollama
    model = ChatOllama(
        cache=True,
        callback_manager=callback_manager,
        model=config["model"],
        repeat_penalty=config["settings"]["repeat_penalty"],
        temperature=config["settings"]["temperature"],
        top_k=config["settings"]["top_k"],
        top_p=config["settings"]["top_p"],
    )
    chain = (
        {
            "context": vectordb.as_retriever(
                search_kwargs={"k": int(config["settings"]["num_sources"])}
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


async def update_cl(config: dict, settings: dict):
    """Update the model configuration.

    Args:
        config (dict): items in config.yaml
        settings (dict): user chat settings input
    """
    if settings:
        config["settings"] = settings
    chain = create_chain(config)
    # https://docs.chainlit.io/api-reference/chat-settings
    inputs = [
        TextInput(
            id="num_sources",
            label="# of Sources",
            initial=str(config["settings"]["num_sources"]),
            description="Number of sources returned based on their similarity score. The same source can be returned more than once. (Default: 4)",
        ),
        Slider(
            id="temperature",
            label="Temperature",
            initial=config["settings"]["temperature"],
            min=0,
            max=1,
            step=0.1,
            description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
        ),
        Slider(
            id="repeat_penalty",
            label="Repeat Penalty",
            initial=config["settings"]["repeat_penalty"],
            min=1.0,
            max=3.0,
            step=0.1,
            description="Sets how strongly to penalize repetitions. A higher value will penalize repetitions more strongly. (Default: 1.1)",
        ),
        Slider(
            id="top_k",
            label="Top K",
            initial=config["settings"]["top_k"],
            min=0,
            max=100,
            step=1,
            description="Reduces the probability of generating nonsense. A higher value will give more diverse answers. (Default: 40)",
        ),
        Slider(
            id="top_p",
            label="Top P",
            initial=config["settings"]["top_p"],
            min=0,
            max=1,
            step=0.1,
            description="Works together with top-k. A higher value will lead to more diverse text. (Default: 0.9)",
        ),
    ]
    cl.user_session.set("chain", chain)

    await cl.ChatSettings(inputs).send()


# https://docs.chainlit.io/integrations/langchain
# https://docs.chainlit.io/examples/qa
@cl.on_chat_start
async def on_chat_start():
    """
    Triggered at the start of a chat session. It loads the model configuration from a file
    and sets it in the user session for future use.
    """
    config = load_config()
    cl.user_session.set("config", config)
    await update_cl(config, {})

    await cl.Message(content=config["introduction"]).send()


@cl.on_message
async def on_message(message: cl.Message):
    "Chat message handler."
    runnable = cl.user_session.get("chain")
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                self.sources.add(d.metadata["source"])  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if self.sources:
                sources_text = "\n".join(self.sources)
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(
            callbacks=[
                cl.LangchainCallbackHandler(),
                PostMessageHandler(msg),
            ],
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()


@cl.on_settings_update
async def setup_agent(settings: dict):
    """Update Chat Settings.

    Args:
        settings (dict): user chat settings input
    """
    config = cl.user_session.get("config")
    await update_cl(config, settings)
