import os
from pathlib import Path
import chainlit as cl
from chainlit.context import init_http_context
from chainlit.input_widget import Slider, TextInput
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from chainlit.server import app
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.cache import SQLiteCache
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.globals import set_llm_cache
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from embed import load_config, parse_args
from utils.api import Query


def setup_memory():
    """Setup memory for the memory of the chat.

    Returns:
        ConversationBufferMemory: buffer for storing conversation memory
    """
    Path("memory").mkdir(parents=True, exist_ok=True)
    # https://python.langchain.com/docs/modules/memory/chat_messages/
    message_history = ChatMessageHistory()
    # https://python.langchain.com/docs/modules/memory/
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    # https://python.langchain.com/docs/integrations/llms/llm_caching
    set_llm_cache(SQLiteCache(database_path="memory/cache.db"))

    return memory


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
        BaseConversationalRetrievalChain: chain for having a conversation based on retrieved documents
    """
    if os.getenv("TEST"):
        config = parse_args(config, ["--test-embed"])
        print("Running in TEST mode.")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    memory = setup_memory()
    vectordb = import_db(config)
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
    # https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
    chain = ConversationalRetrievalChain.from_llm(
        chain_type="stuff",
        llm=model,
        memory=memory,
        retriever=vectordb.as_retriever(
            search_kwargs={"k": int(config["settings"]["num_sources"])}
        ),
        return_source_documents=True,
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
    # https://docs.chainlit.io/observability-iteration/prompt-playground/llm-providers#langchain-provider
    add_llm_provider(
        LangchainGenericProvider(
            id=chain.combine_docs_chain.llm_chain.llm._llm_type,
            name="Ollama",
            llm=chain.combine_docs_chain.llm_chain.llm,
            inputs=[input for input in inputs if isinstance(input, Slider)],
            is_chat=True,
        )
    )
    cl.user_session.set("chain", chain)

    await cl.ChatSettings(inputs).send()


# https://docs.chainlit.io/integrations/langchain
# https://docs.chainlit.io/examples/qa
@cl.on_chat_start
async def on_chat_start():
    "Send a chat start message to the chat and load the model config."
    config = load_config()
    cl.user_session.set("config", config)
    await update_cl(config, None)

    await cl.Message(content=config["introduction"], disable_feedback=True).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle a message.

    Args:
        message (cl.Message): User prompt input
    """
    chain = cl.user_session.get("chain")
    res = await cl.make_async(chain)(
        message.content,
        callbacks=[cl.LangchainCallbackHandler()],
    )
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
        for source_doc in source_documents:
            source_name = source_doc.metadata["source"]
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content,
                    name=source_name,
                )
            )
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


@cl.on_settings_update
async def setup_agent(settings: dict):
    """Update Chat Settings.

    Args:
        settings (dict): user chat settings input
    """
    config = cl.user_session.get("config")
    await update_cl(config, settings)


# http://localhost:8000/docs
@app.get("/ping")
async def ping():
    """Ping the status of the server.

    Returns:
        dict: healthcheck
    """
    return {"status": "Healthy"}


@app.post("/query")
async def prompt(query: Query):
    """Prompt the user for a query.

    Args:
        query (Query): prompt with optional chat settings as FastAPI object

    Returns:
        dict: llm chain response
    """
    init_http_context()
    config = load_config()
    settings = ["num_sources", "temperature", "repeat_penalty", "top_k", "top_p"]
    for setting in settings:
        config["settings"][setting] = query.__dict__[setting]
    chain = create_chain(config)
    message = cl.Message(content=query.prompt)
    res = await cl.make_async(chain)(
        message.content,
        callbacks=[cl.LangchainCallbackHandler()],
    )

    return res
