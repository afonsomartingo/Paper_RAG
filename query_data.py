# from dataclasses import dataclass
import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import tkinter as tk
from tkinter import messagebox

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

class App:
    def __init__(self,root):
        self.root = root
        self.root.title("Scientific Paper Generator and Database Management")

        # Create the widgets for the GUI
        self.api_key_label = tk.Label(root, text="OpenAI API Key:")
        self.api_key_label.pack()
        self.api_key_entry = tk.Entry(root, show="*")
        self.api_key_entry.pack()
        
        self.query_label = tk.Label(root, text="Query Text:")
        self.query_label.pack()
        self.query_entry = tk.Entry(root)
        self.query_entry.pack()
        
        self.query_button = tk.Button(root, text="Query Database", command=self.query_database)
        self.query_button.pack()

        self.result_text = tk.Text(root, height=20, width=80)
        self.result_text.pack()

    def query_database(self):
            api_key = self.api_key_entry.get()
            os.environ["OPENAI_API_KEY"] = api_key

            query_text = self.query_entry.get()
            if not query_text:
                messagebox.showerror("Error", "Please enter a query text.")
                return

            embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
            # Search the DB
            results = db.similarity_search(query_text, k=5)  # Search the database for the most similar documents
            if len(results) == 0:  # If no results are found
                self.result_text.insert(tk.END, "No results found\n")
                return
            
            # Search the DB articles
            results_articles = db.similarity_search(query_text, k=5)  # Search the database for the most similar documents

            context_text = "\n\n---\n\n".join([doc.page_content for doc in results])  # Get the context text
            external_context = "\n\n---\n\n".join([doc.page_content for doc in results_articles])  # Get the context

            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, external_context=external_context, question=query_text)

            model = ChatOpenAI(openai_api_key=api_key)
            response_text = model.invoke(prompt)  # Use invoke instead of predict
            
            sources = [doc.metadata.get("source", None) for doc in results]
            formatted_response = f"Response: {response_text}\nSources: {sources}"
            self.result_text.insert(tk.END, formatted_response + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()