#---------------
# load documents
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.json_loader import JSONLoader


GSM8K_PATH = "./data/gsm8k/" #"./src/RAG/data/gsm8k/"
loader = DirectoryLoader(
    path=GSM8K_PATH, 
    glob="./*.jsonl",
    #,
    show_progress=True,
    loader_cls=JSONLoader, 
    loader_kwargs = {
        'text_content' : False,
        'jq_schema' : '.question, .answer',
        'json_lines' : True,
        },
)
docs = loader.load()


# split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)

#-----------------------------
# create persistente vector db

# name of the futur persistent vector database
persist_directory = "vdb_gsm8k"


# embedding model
# ollama embeddings to vectorize the data
from langchain_community.embeddings import OllamaEmbeddings
embeddings_open = OllamaEmbeddings(model="phi")

#from chromadb.utils import embedding_functions
#embeddings_open = embedding_functions.DefaultEmbeddingFunction()

# create a vector database
from langchain_community.vectorstores import Chroma

print("Starts to embbed chunks ...")
vectordb = Chroma.from_documents(
    documents=texts,
    # Chose the embedding you want to use
    embedding=embeddings_open,
    persist_directory=persist_directory,
)
print("Embedding done :)")

# persist database to the disc to save it
vectordb.persist()
