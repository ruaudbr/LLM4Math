# To load an LLM through Ollama
from langchain_community.embeddings import OllamaEmbeddings
# To create/ use the vector database
from langchain_community.vectorstores import Chroma
# To use the retrieval QA chain
from langchain.chains import RetrievalQA
# To load the LLM through Ollama
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# To enhance the Q&A chain with a more sophisticated prompt template
from langchain.prompts import PromptTemplate


# name of the vector database stored on the disk
#persist_directory = 'vdb_gsm8k'
persist_directory = "vdb_profEnPoche_examples"

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
    model="mixtral",
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
)    


# -----------------
# utils 

#If you want your RAG to also state the sources of 
# the chunks that were returned
def process_llm_response(llm_response):
    print(llm_response['result'])
    all_sources = llm_response["source_documents"]
    
    print("\n" + 20*"-")
    print(f'\nGeneration was augmented with {len(all_sources)} sources:')
    print(10*"-")
    for source in all_sources:
        ## print the source file and idx
        #source_path = source.metadata['source'].split("/")
        #source_path = "/".join(source_path[:-2])
        #print(f"Source file : {source_path}")
        #print(f"Sequence #{source.metadata['seq_num']}")
        #
        #content = source.page_content
        #print(f"Content : {content}\n")
        content = source.page_content
        print(content)
        print(10*"-")

def build_prompt(template_chosen="math_template"):
    math_template = """ You are an assitant to a grade school teacher. Especially, you help create math word problems.
    You always use example relevant to kids and when requested to provided answer, you explain everything step by step.
    Before you anwser the question, take a look at some examples of good questions an answers.
    Examples of good math problems for children: 
    {context}
    
    Please use these examples to answer the following request from the grade school teacher :
    {question}
    
    Your helpful answer based on the request and the provided exmaples:"""

    template_without_context_1 = """You are a passionate school grade teacher. You love teaching kids about maths, history, languages, ... 
    You always use example relevant to kids and explain everything step by step very clearly so that every children can understand.
    Question: {question}
    Helpful Answer:"""

    if template_chosen == "math_template":
        prompt = PromptTemplate(input_variables=["context", "question"], template=math_template)
        return prompt

    elif template_chosen == "template_without_context_1":
        prompt = PromptTemplate(input_variables=["question"], template=template_without_context_1)
        return prompt

    else:
        print(f"""Please choose a valid template :
            - '{math_template}' with context added from the vector database (RAG)
            - '{template_without_context_1}' for the same persona but without the RAG."""
        )



# --------------------- 
# Enhance the Q&A chain:
qa_chain = RetrievalQA.from_chain_type(llm=llm_open,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True,
                                  chain_type_kwargs={"prompt": build_prompt("template_with_context_1")})
        
# -----------------
# testing the RAG


#Write exercises on additions with a dinosaures theme
still_generating = True
while still_generating:
    question = input("Write your prompt/query to the RAG system : \n(to exit type '!exit')\n")
    if question == "!help":
        print("!exit to close the model")
    elif question == "!exit":
        still_generating = False
    else:
        # Question
        llm_response = qa_chain.invoke(question)
        process_llm_response(llm_response)

        