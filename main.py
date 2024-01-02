from chainlit.input_widget import Slider, TextInput
from chainlit.playground.config import add_llm_provider
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.vectorstores import Chroma
from provider import LangchainGenericProvider

import chainlit as cl

settings = {
    "num_sources": 4,
    "repeat_penalty": 1.3,
    "temperature": 0.4,
    "top_k": 20,
    "top_p": 0.35
}

# https://github.com/tylertitsworth/multi-mediawiki-rag#prerequisites
def create_chain():
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    embeddings = HuggingFaceEmbeddings(
        cache_folder="./model",
    )
    vectordb = Chroma(persist_directory="data", embedding_function=embeddings)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model = ChatOllama(
        cache=True,
        callback_manager=callback_manager,
        model="neural-chat",
        repeat_penalty=settings["repeat_penalty"],
        temperature=settings["temperature"],
        top_k=settings["top_k"],
        top_p=settings["top_p"],
    )
    chain = ConversationalRetrievalChain.from_llm(
        chain_type="stuff",
        llm=model,
        memory=memory,
        retriever=vectordb.as_retriever(search_kwargs={"k": int(settings["num_sources"])}),
        return_source_documents=True,
    )

    return chain


async def update_cl(settings):
    chain = create_chain()
    inputs = [
        TextInput(
            id="num_sources",
            label="# of Sources",
            initial=str(settings["num_sources"]),
            description="Number of sources returned based on their similarity score. The same source can be returned more than once. (Default: 4)",
        ),
        Slider(
            id="temperature",
            label="Temperature",
            initial=settings["temperature"],
            min=0,
            max=1,
            step=0.1,
            description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
        ),
        Slider(
            id="repeat_penalty",
            label="Repeat Penalty",
            initial=settings["repeat_penalty"],
            min=0.5,
            max=2.5,
            step=0.1,
            description="Sets how strongly to penalize repetitions. A higher value will penalize repetitions more strongly. (Default: 1.1)",
        ),
        Slider(
            id="top_k",
            label="Top K",
            initial=settings["top_k"],
            min=0,
            max=100,
            step=1,
            description="Reduces the probability of generating nonsense. A higher value will give more diverse answers. (Default: 40)",
        ),
        Slider(
            id="top_p",
            label="Top P",
            initial=settings["top_p"],
            min=0,
            max=1,
            step=0.1,
            description="Works together with top-k. A higher value will lead to more diverse text. (Default: 0.9)",
        ),
    ]
    add_llm_provider(
        LangchainGenericProvider(
            id=chain.combine_docs_chain.llm_chain.llm._llm_type,
            name="Ollama",
            llm=chain.combine_docs_chain.llm_chain.llm,
            is_chat=True,
            inputs=[input for input in inputs if isinstance(input, Slider)],
        )
    )
    await cl.ChatSettings(inputs).send()
    cl.user_session.set("chain", chain)


@cl.on_chat_start
async def on_chat_start():
    await update_cl(settings)


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
    await update_cl(settings)
