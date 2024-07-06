import os
import re
import time
import random
import pickle
import streamlit as st
from typing import List

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Set environment variables
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat message history for a given session ID.

    Args:
        session_id (str): The ID of the session.

    Returns:
        BaseChatMessageHistory: The chat message history for the session.
    """
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]


def extract_text_from_pdf(path_directory: str) -> List[str]:
    """
    Extracts text from PDF files in the specified directory.

    Args:
        path_directory (str): The path to the directory containing the PDF files.

    Returns:
        list: A list of extracted text from the PDF files.
    """
    docs = []
    for file_name in os.listdir(path_directory):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(path_directory, file_name)
            try:
                loader = PyPDFLoader(file_path=pdf_path)
                print(f"Processing: {file_name}")
                doc = loader.load_and_split()
                docs.extend(doc)
                return docs
            except Exception as e:
                st.error(f"Error processing {pdf_path}: {e}")
                return []


def get_text_chunks(docs: List[str]) -> List[str]:
    """
    Splits a list of documents into smaller text chunks.

    Args:
        docs (list): A list of documents to be split.

    Returns:
        list: A list of text chunks obtained by splitting the documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    return splits


def get_vectordb(splits: List[str], path_directory: str) -> Chroma:
    """
    Generates a vectorstore from a list of document splits.

    Args:
        splits (list): A list of document splits.
        path_directory (str): The path to the directory where the vectorstore will be saved.

    Returns:
        Chroma: The generated vectorstore.
    """
    print("Now generating vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=f"{path_directory}/vectordb",
    )
    return vectorstore


def filter_relevant_chunks(query, vectordb, k=5):
    query_embedding = OpenAIEmbeddings().embed_query(query)
    results = vectordb.similarity_search(query_embedding, k=k)
    return results


def conversational_rag(vectordb: Chroma) -> RunnableWithMessageHistory:
    """
    Given a vector database, this function creates a retrieval chain for conversational question-answering tasks.
    The retrieval chain is designed to handle chat history and the latest user question, and formulate a standalone question
    that can be understood without the chat history. The function uses a contextualize question prompt and a question-answer
    prompt to interact with the user and retrieve relevant context for answering the question.

    Parameters:
    - vectordb: The vector database used for retrieval.

    Returns:
    - rag_chain: The retrieval chain for conversational question-answering tasks.
    """
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    system_prompt = (
        "You are an assistant for question-answering tasks who responds in markdown. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def store_history(rag_chain: RunnableWithMessageHistory) -> RunnableWithMessageHistory:
    """
    Stores the history of a conversational RAG chain.

    Args:
        rag_chain (RunnableWithMessageHistory): The RAG chain to store the history for.

    Returns:
        RunnableWithMessageHistory: The conversational RAG chain with history stored.
    """
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def save_uploaded_file(pdf_doc) -> None:
    """
    Save the uploaded PDF document to the 'data' directory.

    Args:
        pdf_doc (file-like object): The PDF document to be saved.

    Returns:
        None
    """
    os.makedirs("data/pdfs", exist_ok=True)
    if os.path.exists("data/pdfs"):
        for file_name in os.listdir("data/pdfs/"):
            file_path = os.path.join("data/pdfs/", file_name)
            os.remove(file_path)
    with open(f"data/pdfs/{pdf_doc.name}", "wb") as f:
        f.write(pdf_doc.getbuffer())


def verify_answer_with_documents(answer, documents):
    answer_keywords = set(answer.lower().split())
    total_keywords = len(answer_keywords)

    if total_keywords == 0:
        return False, 0.0

    max_overlap = 0

    for doc in documents:
        doc_text = doc.page_content.lower()
        doc_keywords = set(doc_text.split())
        overlap = len(answer_keywords.intersection(doc_keywords))
        max_overlap = max(max_overlap, overlap)

    confidence_score = max_overlap / total_keywords
    verification_status = confidence_score > 0.0

    return verification_status, confidence_score


def display_answer_with_references(
    answer, references, confidence_score, verifiable_status
):
    st.markdown(
        f"### Answer (Confidence: {confidence_score*100:.2f}%), Verifiable: {verifiable_status}:"
    )
    with st.chat_message("assistant"):
        response = st.write_stream(stream(answer))
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("#### References:")
    for ref in references:
        st.markdown(f"- {ref.page_content}")
        st.markdown(f"- Page: {ref.metadata['page']}")
        st.markdown(f"- Source: {ref.metadata['source']}")
        st.markdown("---")

    if confidence_score < 0.5:
        st.markdown(
            "**Note:** The confidence score is low. The answer might not be accurate."
        )


def generate_faqs(conversational_rag_chain):
    faqs = []
    companies = ["apple", "amazon", "meta", "microsoft", "google"]
    common_terms = ["revenue", "profit", "loss", "expenses", "income", "earnings"]
    years = ["2020", "2021", "2022"]
    combinations = [
        (company, term, year)
        for company in companies
        for term in common_terms
        for year in years
    ]

    common_terms = random.sample(combinations, 4)
    for term in common_terms:
        question = f"What is {term[0]}'s {term[1]} in {term[2]}?"
        answer = conversational_rag_chain.invoke(
            {"input": question}, config={"configurable": {"session_id": "1"}}
        )["answer"]
        faqs.append((question, answer))
    return faqs


def stream(text):
    for word in text.split()[:100]:
        yield word + " "
        time.sleep(0.05)


def main():
    """
    Main function that sets up the chatbot application and handles user interactions.

    The function sets the page configuration, displays the menu, allows users to upload PDF files,
    processes the PDFs, and generates answers to user questions using a conversational RAG model.
    """
    st.set_page_config(
        page_title="Chatbot that answers questions related to Financial Reports of Companies!",
        page_icon=":file_pdf:",
    )

    st.markdown(
        "<h2 style='color: #D81B60; text-align: center;'>(FAANG) Financial Reports Chatbot</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h2 style='color: #3498db; text-align: center;'>Powered by Mixtral-8x7B-Instruct-v0.1!</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='color: #ddd;'>Ask a question to the chatbot ≽^•⩊•^≼</h3>",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # generate a random hash for the session_id
    st.session_state["session_id"] = re.sub(r"\W+", "", str(random.getrandbits(32)))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.markdown("<h3 style='color: #2ecc71;'>Menu:</h3>", unsafe_allow_html=True)
        # company_name = st.text_input("**Company name**", key="company_input")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files =>", accept_multiple_files=True
        )
        if pdf_docs:  # and company_name:
            for pdf_doc in pdf_docs:
                save_uploaded_file(pdf_doc)  # , company_name=company_name)
            st.write("Uploaded and saved files.")

        if st.button(
            "Process PDFs",
        ):
            with st.spinner("Processing..."):
                # if os.path.exists("data/vectordb"):
                #     vectordb = Chroma(
                #         persist_directory="data/vectordb",
                #         embedding_function=OpenAIEmbeddings(),
                #     )
                # else:
                docs = extract_text_from_pdf("data/pdfs/")
                splits = get_text_chunks(docs)
                vectordb = get_vectordb(splits, "data/")

                # if not os.path.exists("data/docs.pkl"):
                #     docs = extract_text_from_pdf("data/pdfs/")
                #     with open("data/docs.pkl", "wb") as f:
                #         pickle.dump(docs, f)
                # else:
                # with open("data/docs.pkl", "rb") as f:
                #     docs = pickle.load(f)
                st.write("PDFs processed and indexed.")
                st.session_state["docs"] = docs
                st.session_state["vectordb"] = vectordb

    if user_question := st.chat_input("Enter your question here:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if "rag_chain" not in st.session_state:
            with st.spinner("Loading chatbot ..."):
                vectordb = st.session_state.get("vectordb")
                if vectordb is None:
                    st.error("Please process PDFs before asking questions.")
                    return
                rag_chain = conversational_rag(vectordb)
                st.session_state["rag_chain"] = store_history(rag_chain)

        conversational_rag_chain = st.session_state["rag_chain"]
        with st.spinner("Generating answer ..."):
            response = conversational_rag_chain.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": st.session_state["session_id"]}},
            )

            answer = response["answer"]
            references = response.get("context", [])

            verifiable_status, confidence_score = verify_answer_with_documents(
                answer, references
            )
            display_answer_with_references(
                answer, references, confidence_score, verifiable_status
            )

        with st.sidebar:
            if st.session_state.get("docs"):
                st.markdown(
                    "<h3 style='color: #D81B60;'>FAQs:</h3>", unsafe_allow_html=True
                )
                faqs = generate_faqs(conversational_rag_chain)
                for question, answer in faqs:
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")


if __name__ == "__main__":
    main()
