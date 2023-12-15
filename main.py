from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import MWDumpLoader
from langchain.document_loaders.merge import MergedDataLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import chainlit as cl

# from dotenv import load_dotenv
# load_dotenv()

def create_vector_db():
    # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./model")

    wikis = {
        "dungeons": "",
        "eberron": "",
        "forgottenrealms": "",
        "planescape": "",
    }
    for wiki in wikis.keys():
        # https://python.langchain.com/docs/integrations/document_loaders/mediawikidump
        wikis[wiki] = MWDumpLoader(
            file_path=f"sources/{wiki}_pages_current.xml",
            encoding="utf-8",
            skip_redirects=True,
            stop_on_error=False
        )
    # https://python.langchain.com/docs/integrations/document_loaders/merge_doc
    loader_all = MergedDataLoader(loaders=wikis.values())
    # https://python.langchain.com/docs/modules/data_connection/document_transformers/#get-started-with-text-splitters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(loader_all.load())
    # https://python.langchain.com/docs/integrations/vectorstores/chroma
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="data",
        # ids=[str(i) for i in range(len(loader_all.load()))],
    )
    vectordb.persist()

def create_chain():
    system_prompt="""
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
----
Content: {context}
---
"""
    human_prompt = "Question: ```{question}```"
    # https://python.langchain.com/docs/modules/memory/chat_messages/
    message_history = ChatMessageHistory()
    # https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ]
    )
    # https://python.langchain.com/docs/modules/memory/
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./model"
    )
    vectordb = Chroma(persist_directory="data", embedding_function=embeddings)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # https://python.langchain.com/docs/integrations/llms/huggingface_pipelines
    model = HuggingFacePipeline.from_model_id(
        model="Intel/neural-chat-7b-v3-1",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
        verbose=False,
        callback_manager=callback_manager,
    )
    # https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# https://docs.chainlit.io/integrations/langchain
# https://docs.chainlit.io/examples/qa
@cl.on_chat_start
async def on_chat_start():
    chain = create_chain()
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


if __name__ == "__main__":
    create_vector_db()
    chain = create_chain()
    res = chain("List every octopus monster in the forgotten realms")
    answer = res["answer"]
    print([source_doc.page_content for source_doc in res["source_documents"]])
