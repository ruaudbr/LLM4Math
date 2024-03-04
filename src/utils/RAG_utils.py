from langchain.prompts import PromptTemplate

# -----------------
# utils


# If you want your RAG to also state the sources of
# the chunks that were returned
def process_llm_response(llm_response):
    print(llm_response)
    all_sources = llm_response["source_documents"]
    result = (
        f"\nGeneration was augmented with {len(all_sources)} sources:\n"
        + 10 * "-"
        + "\n"
    )
    for source in all_sources:
        ## print the source file and idx
        # source_path = source.metadata['source'].split("/")
        # source_path = "/".join(source_path[:-2])
        # print(f"Source file : {source_path}")
        # print(f"Sequence #{source.metadata['seq_num']}")
        #
        # content = source.page_content
        # print(f"Content : {content}\n")
        result += source.page_content + "\n" + 10 * "-" + "\n"
    result += 20 * "=" + "\n"
    return result + llm_response["result"]


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
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=math_template
        )
        return prompt

    elif template_chosen == "template_without_context_1":
        prompt = PromptTemplate(
            input_variables=["question"], template=template_without_context_1
        )
        return prompt

    else:
        print(
            f"""Please choose a valid template :
            - '{math_template}' with context added from the vector database (RAG)
            - '{template_without_context_1}' for the same persona but without the RAG."""
        )
