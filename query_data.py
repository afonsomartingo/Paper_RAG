import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import tkinter as tk
from tkinter import ttk, messagebox
from concurrent.futures import ThreadPoolExecutor

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
2. Include relevant analysis or discussion based on the context.
3. Be well-structured and coherent.  

Specific topic: {question}

---

Write the scientific text based on the provided context and follow all the guidelines given at the Specific topic:
"""

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Scientific Paper Generator")
        self.root.geometry("800x600")
        self.root.configure(bg="#2e2e2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 12), background="#2e2e2e", foreground="white")
        style.configure("TButton", font=("Helvetica", 12), background="#4e4e4e", foreground="white")
        style.configure("TEntry", font=("Helvetica", 12), fieldbackground="#4e4e4e", foreground="white")
        style.configure("TFrame", background="#2e2e2e")

        self.result_frame = ttk.Frame(root)
        self.result_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(self.result_frame, height=20, width=80, wrap=tk.WORD, font=("Helvetica", 12), bg="#4e4e4e", fg="white")
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=self.scrollbar.set)

        self.query_label = ttk.Label(root, text="Query Text:")
        self.query_label.pack(pady=10, side=tk.BOTTOM)
        self.query_entry = ttk.Entry(root)
        self.query_entry.pack(pady=5, padx=20, fill=tk.X, side=tk.BOTTOM)

        self.api_key_label = ttk.Label(root, text="OpenAI API Key:")
        self.api_key_label.pack(pady=10, side=tk.BOTTOM)
        self.api_key_entry = ttk.Entry(root, show="*")
        self.api_key_entry.pack(pady=5, padx=20, fill=tk.X, side=tk.BOTTOM)

        self.query_button = ttk.Button(root, text="Query Database", command=self.query_database)
        self.query_button.pack(pady=20, side=tk.BOTTOM)

    def query_database(self):
        api_key = self.api_key_entry.get()
        os.environ["OPENAI_API_KEY"] = api_key

        query_text = self.query_entry.get()
        if not query_text:
            messagebox.showerror("Error", "Please enter a query text.")
            return

        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            future_results = executor.submit(db.similarity_search, query_text, k=5)
            results = future_results.result()

        if len(results) == 0:  # If no results are found
            self.result_text.insert(tk.END, "No results found\n")
            return
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            future_results_articles = executor.submit(db.similarity_search, query_text, k=5)
            results_articles = future_results_articles.result()

        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])  # Get the context text
        external_context = "\n\n---\n\n".join([doc.page_content for doc in results_articles])  # Get the context

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, external_context=external_context, question=query_text)

        model = ChatOpenAI(openai_api_key=api_key)
        response = model.invoke(prompt, max_tokens=500)  # Use invoke with max_tokens
        
        response_text = response['choices'][0]['message']['content'].strip()  # Extract only the content
        
        self.result_text.insert(tk.END, f"{response_text}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()