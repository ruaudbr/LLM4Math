# load documents
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="./data/gsm8k/", glob="./*.jsonl", text_content=False, json_lines=True
)
docs = loader.load()


# split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)


# name of the futur persistent vector database
persist_directory = "vdb_gsm8k"

# ollama embeddings to vectorize the data
from langchain.embeddings import OllamaEmbeddings

embeddings_open = OllamaEmbeddings(model="phi")

# create a vector database
from langchain.vectorstores import Chroma

vectordb = Chroma.from_documents(
    documents=texts,
    # Chose the embedding you want to use
    embedding=embeddings_open,
    persist_directory=persist_directory,
)

# persist database to the disc to save it
vectordb.persist()
