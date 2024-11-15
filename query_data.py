import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Using exclusively the following contexts:

1. Primary context (your thesis document):
{context}

2. Supplemental context (scientific articles from top-tier conferences that you need to follow the structure as reference):
{external_context}

---

Compose a scientifically structured text on the topic below, suitable for submission to top-tier scientific conferences. The text should:

1. Clearly and accurately explain the concept or technique.
3. Include relevant analysis or discussion based on the context.
4. Be well-structured and coherent.  

Specific topic: {question}

---

Write the scientific text based on the provided context and follow all the guidelines given at the Specific topic:
"""


def main():
    # Create CLI
    parser = argparse.ArgumentParser(description="Query the database")
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()  # Parse the arguments
    query_text = args.query_text  # Get the query text

    # Prompt the user for the API key
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")

    # Use the API key in your code
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Prepare the DB
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory="chroma", embedding_function=embedding_function)
    
    # Search the DB
    results = db.similarity_search(query_text, k=5)  # Search the database for the most similar documents
    if len(results) == 0:  # If no results are found
        print("No results found")  # Print "No results found"
        return
    
    # Search the DB articles
    results_articles = db.similarity_search(query_text, k=5)  # Search the database for the most similar documents

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])  # Get the context text
    external_context = "\n\n---\n\n".join([doc.page_content for doc in results_articles])  # Get the context

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, external_context=external_context, question=query_text)

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)  # Use invoke instead of predict
    
    sources = [doc.metadata.get("source", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()