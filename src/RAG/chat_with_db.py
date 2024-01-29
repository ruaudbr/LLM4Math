# To load an LLM through Ollama
from langchain.embeddings import OllamaEmbeddings
# To create/ use the vector database
from langchain.vectorstores import Chroma
# To use the retrieval QA chain
from langchain.chains import RetrievalQA
# To load the LLM through Ollama
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# To enhance the Q&A chain with a more sophisticated prompt template
from langchain.prompts import PromptTemplate


# name of the vector database stored on the disk
persist_directory = 'vdb_gsm8k'
# ollama embeddings used to vectorize the data
embeddings_open = OllamaEmbeddings(model="phi")

# load the vector database from disk
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings_open,
)
# instanciate a retriever to fetch the most similar vectors in the vdb
retriever = vectordb.as_retriever()





# -----------------
# LLM

llm_open = Ollama(
    model="mistral",
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
)    
qa_chain = RetrievalQA.from_chain_type(llm=llm_open,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True)


# -----------------
# utils 

#If you want your RAG to also state the sources of 
# the chunks that were returned
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


def build_prompt(template_num="template_1"):
    template = """ You are a helpful chatbot, named RSLT. You answer the questions of the customers giving a lot of details based on what you find in the context.
Do not say anything that is not in the website
You are to act as though you're having a conversation with a human.
You are only able to answer questions, guide and assist, and provide recommendations to users. You cannot perform any other tasks outside of this.
Your tone should be professional and friendly.
Your purpose is to answer questions people might have, however if the question is unethical you can choose not to answer it.
Your responses should always be one paragraph long or less.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    template2 = """You are a helpful chatbot, named RSLT. You answer the questions of the customers giving a lot of details based on what you find in the context. 
    Your responses should always be one paragraph long or less.
    Question: {question}
    Helpful Answer:"""

    if template_num == "template_1":
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        return prompt

    elif template_num == "template_2":
        prompt = PromptTemplate(input_variables=["question"], template=template2)
        return prompt

    else:
        print("Please choose a valid template")
 

# -----------------
# testing the RAG

# Question
query = "What is this document about?"
llm_response = qa_chain(query)
process_llm_response(llm_response)


       
        
# Enhance the Q&A chain:
qa_chain = RetrievalQA.from_chain_type(llm=llm_open,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True,
                                  chain_type_kwargs={"prompt": build_prompt("template_1")})