from chainlit.input_widget import Slider
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.cache import SQLiteCache
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.document_loaders import MWDumpLoader
from langchain.document_loaders.merge import MergedDataLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.globals import set_llm_cache
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.vectorstores import Chroma
from sys import exit

import argparse
import chainlit as cl
import yaml


class MultiWiki:
    def __init__(self):
        try:
            with open("config.yaml", "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except FileNotFoundError:
            print("Error: File config.yaml not found.")
            exit(1)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            exit(1)

        for key, val in data.items():
            if key == "mediawikis":
                self.wikis = {wiki: "" for wiki in data["mediawikis"]}
            else:
                setattr(self, key, val)

    def set_args(self, args):
        self.args = args

    # https://github.com/jmorganca/ollama/blob/main/docs/api.md#show-model-information
    def set_chat_settings(self, settings):
        if not settings:
            self.inputs = wiki.settings
        else:
            self.inputs = settings


### Globals
wiki = MultiWiki()


def create_vector_db(embeddings_model, source, wikis):
    if not source:
        print("No data sources found")
        exit(1)

    # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model, cache_folder="./model"
    )

    for wiki in wikis.keys():
        # https://python.langchain.com/docs/integrations/document_loaders/mediawikidump
        wikis[wiki] = MWDumpLoader(
            encoding="utf-8",
            file_path=f"{source}/{wiki}_pages_current.xml",
            # https://www.mediawiki.org/wiki/Help:Namespaces
            namespaces=[0],
            skip_redirects=True,
            stop_on_error=False,
        )
    # https://python.langchain.com/docs/integrations/document_loaders/merge_doc
    loader_all = MergedDataLoader(loaders=wikis.values())
    documents = loader_all.load()
    print(f"Embedding {len(documents)} Pages, this may take a while.")
    # https://python.langchain.com/docs/integrations/vectorstores/chroma
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="data",
    )
    vectordb.persist()


def create_chain(embeddings_model, model):
    # https://python.langchain.com/docs/modules/memory/chat_messages/
    message_history = ChatMessageHistory()
    # https://python.langchain.com/docs/modules/memory/
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    embeddings = HuggingFaceEmbeddings(
        cache_folder="./model",
        model_name=embeddings_model,
    )
    vectordb = Chroma(persist_directory="data", embedding_function=embeddings)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # https://python.langchain.com/docs/integrations/llms/llm_caching
    set_llm_cache(SQLiteCache(database_path="memory/cache.db"))
    wiki.set_chat_settings(None)
    # https://python.langchain.com/docs/integrations/llms/ollama
    model = ChatOllama(
        cache=True,
        callback_manager=callback_manager,
        model=model,
        repeat_penalty=wiki.inputs["repeat_penalty"],
        temperature=wiki.inputs["temperature"],
        top_k=wiki.inputs["top_k"],
        top_p=wiki.inputs["top_p"],
    )
    # https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
    chain = ConversationalRetrievalChain.from_llm(
        chain_type="stuff",
        llm=model,
        memory=memory,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
    )

    return chain, model


# https://docs.chainlit.io/integrations/langchain
# https://docs.chainlit.io/examples/qa
@cl.on_chat_start
async def on_chat_start():
    chain, llm = create_chain(
        wiki.embeddings_model,
        wiki.model,
    )

    # https://docs.chainlit.io/api-reference/chat-settings
    inputs = [
        Slider(
            id="temperature",
            label="Temperature",
            initial=wiki.inputs["temperature"],
            min=0,
            max=1,
            step=0.1,
            description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
        ),
        Slider(
            id="repeat_penalty",
            label="Repeat Penalty",
            initial=wiki.inputs["repeat_penalty"],
            min=0.5,
            max=2.5,
            step=0.1,
            description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
        ),
        Slider(
            id="top_k",
            label="Top K",
            initial=wiki.inputs["top_k"],
            min=0,
            max=100,
            step=1,
            description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
        ),
        Slider(
            id="top_p",
            label="Top P",
            initial=wiki.inputs["top_p"],
            min=0,
            max=1,
            step=0.1,
            description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
        ),
    ]

    # https://docs.chainlit.io/observability-iteration/prompt-playground/llm-providers#langchain-provider
    add_llm_provider(
        LangchainGenericProvider(
            id=llm._llm_type,
            name="Ollama",
            llm=llm,
            is_chat=True,
            # Not enough context to LangchainGenericProvider
            # https://github.com/Chainlit/chainlit/blob/main/backend/chainlit/playground/providers/langchain.py#L27
            # inputs=inputs,
        )
    )
    await cl.ChatSettings(inputs).send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = await cl.make_async(chain)(
        message.content,
        callbacks=[cl.LangchainCallbackHandler()],
    )
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = source_doc.metadata["source"]
            # Create the text element referenced in the message
            if source_idx == 0 or source_name != source_documents[0].metadata["source"]:
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            else:
                continue
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


@cl.on_settings_update
async def setup_agent(settings):
    wiki.set_chat_settings(settings)
    await on_chat_start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", dest="embed", action="store_false")

    wiki.set_args(parser.parse_args())

    if wiki.args.embed:
        create_vector_db(
            wiki.embeddings_model,
            wiki.source,
            wiki.wikis,
        )

    chain, llm = create_chain(
        wiki.embeddings_model,
        wiki.model,
    )

    if not wiki.question:
        print("No Prompt for Chatbot found")
        exit(1)

    res = chain(wiki.question)
    answer = res["answer"]
    print(answer)
    print([source_doc.page_content for source_doc in res["source_documents"]])
