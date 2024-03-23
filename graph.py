from functools import partial
from typing import Dict, TypedDict

# import chainlit as cl
from langchain import hub

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


def retrieve(state: dict, retriever: VectorStoreRetriever):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.get_relevant_documents(question)

    return {"keys": {"documents": documents, "question": question}}


def generate(state: dict, model: ChatOllama):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    # with cl.Step(name="Generate") as step:
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    setattr(model, "format", None)
    # Chain
    rag_chain = prompt | model | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    print(generation)
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


def grade_documents(state: dict, model: ChatOllama):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # LLM
    setattr(model, "format", "json")

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question", "context"],
    )

    # Chain
    chain = prompt | model | JsonOutputParser()

    # Score
    filtered_docs = []
    for _d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": _d.page_content,
            }
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(_d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"keys": {"documents": filtered_docs, "question": question}}


def transform_query(state: dict, model: ChatOllama):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n
        Look at the input and try to reason about the underlying sematic intent / meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question:""",
        input_variables=["question"],
    )
    setattr(model, "format", None)
    # Chain
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print(better_question)

    return {"keys": {"documents": documents, "question": better_question}}


def prepare_for_final_grade(state: dict):
    """
    Passthrough state for final grade.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The current graph state
    """

    print("---FINAL GRADE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]

    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


def decide_to_generate(state: dict):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    # question = state_dict["question"]
    filtered_documents = state_dict["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents(state: dict, model: ChatOllama):
    """
    Determines whether the generation is grounded in the document.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs DOCUMENTS---")
    state_dict = state["keys"]
    # question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]

    # LLM
    setattr(model, "format", "json")

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n
        Here are the facts:
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["generation", "documents"],
    )

    # Chain
    chain = prompt | model | JsonOutputParser()
    score = chain.invoke({"generation": generation, "documents": documents})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: SUPPORTED, MOVE TO FINAL GRADE---")
        return "supported"
    else:
        print("---DECISION: NOT SUPPORTED, GENERATE AGAIN---")
        return "not supported"


def grade_generation_v_question(state: dict, model: ChatOllama):
    """
    Determines whether the generation addresses the question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs QUESTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    # documents = state_dict["documents"]
    generation = state_dict["generation"]

    setattr(model, "format", "json")

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n
        Here is the answer:
        \n ------- \n
        {generation}
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["generation", "question"],
    )

    # Prompt
    chain = prompt | model | JsonOutputParser()
    score = chain.invoke({"generation": generation, "question": question})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: USEFUL---")
        return "useful"
    else:
        print("---DECISION: NOT USEFUL---")
        return "not useful"


def create_workflow(
    config: dict, retriever: VectorStoreRetriever  # , memory: ConversationBufferMemory
):
    """Creates a workflow graph state machine for a self-correcting RAG.

    Args:
        config (dict): items in config.yaml
        retriever (VectorStoreRetriever): vector database retriever

    Returns:
        CompiledGraph: compiled workflow graph
    """
    workflow = StateGraph(GraphState)

    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model = ChatOllama(
        cache=True,
        # callback_manager=callback_manager,
        model=config["model"],
        # repeat_penalty=config["settings"]["repeat_penalty"],
        temperature=config["settings"]["temperature"],
        # top_k=config["settings"]["top_k"],
        # top_p=config["settings"]["top_p"],
    )

    workflow.add_node("retrieve", partial(retrieve, retriever=retriever))
    workflow.add_node("grade_documents", partial(grade_documents, model=model))
    workflow.add_node("generate", partial(generate, model=model))
    workflow.add_node("transform_query", partial(transform_query, model=model))
    workflow.add_node("prepare_for_final_grade", prepare_for_final_grade)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        partial(grade_generation_v_documents, model=model),
        {
            "supported": "prepare_for_final_grade",
            "not supported": "generate",
        },
    )
    workflow.add_conditional_edges(
        "prepare_for_final_grade",
        partial(grade_generation_v_question, model=model),
        {
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()
