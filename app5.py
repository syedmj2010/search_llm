import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.agents import initialize_agent, Tool, AgentType
import streamlit as st
from sqlalchemy import create_engine, text

# Streamlit page configuration - must be the first Streamlit command
st.set_page_config(page_title="Java Code API Finder", layout="wide")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Folder Path and Query Inputs
folder_path = st.text_input("Enter the folder path containing Java files:", r'C:\Workspace\', key="folder_path")
query = st.text_input("Enter your query related to the Java code:", "Can you provide the api for user creation?", key="query")

# Check if embeddings are already inserted
def embeddings_exist():
    CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/vector_db"
    engine = create_engine(CONNECTION_STRING)
    connection = engine.connect()
    try:
        # Correct SQL query to check if the table exists
        result = connection.execute(text("SELECT to_regclass('public.java_code_vectors');")).fetchone()
        return result[0] is not None  # Returns True if the table exists
    finally:
        connection.close()

# Define Tools
def load_documents(folder_path):
    file_paths = glob.glob(f"{folder_path}\\**\\*.java", recursive=True)
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    return documents

def split_texts(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    texts = text_splitter.split_documents(documents)
    return texts

def get_embeddings_and_search(query, texts):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/vector_db"
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=texts,
        collection_name='java_code_vectors',
        connection_string=CONNECTION_STRING,
    )
    return db.similarity_search_with_score(query, k=2)

def format_java_code(similar_results):
    context = ""
    if isinstance(similar_results, list):
        for doc, score in similar_results:
            if hasattr(doc, 'page_content'):
                context += doc.page_content + "\n\n"
        if context:
            formatted_code = f"```java\n{context}\n```"
            return formatted_code
    return "No relevant code found."

def generate_response(query, context):
    prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}"
    return llm.invoke(prompt)

# Define the tools
tools = [
    Tool(
        name="Document Loader",
        func=load_documents,
        description="Load all Java files from the given folder."
    ),
    Tool(
        name="Text Splitter",
        func=split_texts,
        description="Split documents into text chunks."
    ),
    Tool(
        name="Embedding Search",
        func=get_embeddings_and_search,
        description="Generate embeddings and perform a similarity search."
    ),
    Tool(
        name="Format Java Code",
        func=format_java_code,
        description="Format the returned Java code snippet."
    ),
    Tool(
        name="Generate Model Response",
        func=generate_response,
        description="Generate a response based on the given query and context."
    ),
]

# Initialize the agent
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Streamlit page configuration
st.title("Java Code API Finder")

# Process and display results when the query is entered
if query and folder_path:
    if st.button("Process Query"):
        try:
            # Step 1: Check if embeddings already exist
            if not embeddings_exist():
                # Step 2: Load documents from the folder
                documents = load_documents(folder_path)

                # Step 3: Split the documents into chunks
                texts = split_texts(documents)

                # Step 4: Perform similarity search with the query
                similar_results = get_embeddings_and_search(query, texts)

                # Step 5: Format Java code and get the context
                formatted_code = format_java_code(similar_results)

                # Step 6: Display formatted Java code only once
                if formatted_code != "No relevant code found.":
                    st.markdown("### Found Java Code:")
                    st.markdown(formatted_code)  # Display formatted Java code

                    # Step 7: Generate the model's response based on the query and formatted code
                    response = generate_response(query, formatted_code)
                    st.markdown("### Model Response:")
                    st.write(response)  # Display model's response

                else:
                    st.warning("No relevant code found.")
            else:
                st.info("Embeddings already exist in the database. Skipping embedding insertion.")
                
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("Please provide both a folder path and a query.")
