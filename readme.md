# Scientific Paper Generator and Database Management

This project provides a framework for managing scientific document databases and generating structured scientific texts. It leverages OpenAI models and the LangChain framework to streamline querying, organizing, and utilizing research materials effectively.

---

## **Features**
1. **Document Parsing and Storage:** Extracts content from PDFs and stores them in a searchable vector database.  
2. **Contextual Querying:** Uses OpenAI embeddings to find relevant documents for a given query.  
3. **Scientific Text Generation:** Creates well-structured, context-based scientific texts ready for submission to top-tier conferences.  
4. **Flexible API Integration:** Incorporates OpenAI API for embedding and text generation tasks.  

---

## **Installation**
1. **Clone the Repository**  
    ```bash
    git clone <repository-url>
    cd Paper_RAG
    ```

2. **Install Dependencies**  
    Ensure Python 3.8+ is installed. Then, run:  
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Your OpenAI API Key**  
    During runtime, the script will prompt you to input your OpenAI API key. Alternatively, set it as an environment variable:  
    ```bash
    export OPENAI_API_KEY="your-api-key"
    ```

---

## **Usage**

1. **Preparing the Database**  
    To process documents and populate the vector database:  
    ```bash
    python create_database.py
    ```  
    Place the following documents in their respective directories:  
    - Primary context documents: `data/pdf/`  
    - Supplemental scientific articles: `data/articles/`

2. **Querying the Database**  
    Run the script with your query:  
    ```bash
    python query_database.py "Your research question or topic"
    ```  
    The script will:
    - Search the database for relevant documents.
    - Generate a scientific text based on the found documents and the provided query.

---

## **Code Structure**
1. **`create_database.py`:** Handles document loading, splitting, and storage into the vector database.  
2. **`query_database.py`:** Facilitates querying the database and generating responses using OpenAI models.  
3. **Utility Functions:**  
    - `load_documents`: Loads PDF documents.  
    - `split_text`: Splits documents into manageable chunks for processing.  
    - `save_to_chroma`: Saves document embeddings to the Chroma vector database.  

---

## **Prompt Template**
The generated scientific text adheres to the following template:

> Using exclusively the following contexts:
> 
> Primary context (your thesis document): {context}
> 
> Supplemental context (scientific articles from top-tier conferences): {external_context}
> 
> Compose a scientifically structured text on the topic below, suitable for submission to top-tier scientific conferences. The text should:
> 
> 1. Clearly and accurately explain the concept or technique.
> 2. Include relevant analysis or discussion based on the context.
> 3. Be well-structured and coherent.
> 
> Specific topic: {question}

---

## **Example Workflow**
1. **Populate the directories with relevant PDFs.**  
2. **Create the database:**  
    ```bash
    python create_database.py
    ```  
3. **Query for a topic:**  
    ```bash
    python query_database.py "Explain the implications of quantum computing in AI."
    ```

---

## **Requirements**
1. **Python 3.8+**  
2. **Libraries:** See `requirements.txt`. Includes:  
    - `langchain_community`  
    - `langchain_openai`  
    - `PyPDFLoader`  
    - `Chroma`  

---

## **Troubleshooting**
1. **Permission Errors:** Ensure you have sufficient permissions to delete or create directories.  
2. **Corrupt PDF Files:** Validate your PDFs manually if they fail to load.  
3. **API Issues:** Check your OpenAI API key and rate limits.  

---

## **Future Enhancements**
1. Add support for non-PDF formats.  
2. Improve error handling for database operations.  
3. Extend the prompt template for different scientific disciplines.  

---

**Author:** Afonso Carvalho  
**License:** MIT
