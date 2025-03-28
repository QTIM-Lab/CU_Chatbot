from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and split it into chunks"""
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    return all_splits

def get_embeddings_function(model: str) -> callable:
    """Get an embeddings function"""
    if model == "llama3.2":
        return OllamaEmbeddings(model="llama3.2")
    elif model == 'got-4o':
        return OpenAIEmbeddings(model="text-embedding-3-large")
    else:
        raise ValueError(f"Model {model} not supported")

def create_chroma_db_from_file(file_path: str, model: str, collection_name: str) -> Chroma:
    """Create a Chroma database from a file"""
    documents = load_pdf(file_path)
    return create_chroma_db(collection_name, documents, model)

def create_chroma_db(collection_name: str, documents: list[Document], model: str) -> Chroma:
    """Create a Chroma database"""
    embeddings = get_embeddings_function(model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    vector_store.add_documents(documents)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

def get_context_for_query(query: str, vector_store: Chroma) -> str:
    """Get the context for a query"""
    relevant_documents = vector_store.invoke(query)
    # combine the documents into a single string
    context = " ".join([doc.page_content for doc in relevant_documents])
    return context

if __name__ == "__main__":
    file_path = "RFA_HIV.pdf"
    model_name = "llama3.2"

    # create a Chroma database and add documents to it
    vector_store = create_chroma_db_from_file(file_path, model_name)

    # query the database
    query = "What is the purpose of the RFA?"

    # get ollama model
    model = OllamaModel(model_name, vector_store, PROMPT)

    while query != "exit":
        # get response
        response = model(query)

        print(response)
        query = input("Enter a question (exit to quit): ")