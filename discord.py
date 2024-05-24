import chainlit as cl
from embed import load_config
from app import create_chain


async def update_cl(config: dict, settings: dict):
    """Update the model configuration.

    Args:
        config (dict): items in config.yaml
        settings (dict): user chat settings input
    """
    if settings:
        config["settings"] = settings
    chain = create_chain(config)
    cl.user_session.set("chain", chain)


# https://docs.chainlit.io/integrations/langchain
# https://docs.chainlit.io/examples/qa
@cl.on_chat_start
async def on_chat_start():
    "Send a chat start message to the chat and load the model config."
    config = load_config()
    cl.user_session.set("config", config)
    await update_cl(config, None)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle a message.

    Args:
        message (cl.Message): User prompt input
    """
    await cl.Message(content="Hmmmmm").send()
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

    await cl.Message(
        content=f"{cl.user_session.get('user')}{answer}", elements=text_elements
    ).send()
