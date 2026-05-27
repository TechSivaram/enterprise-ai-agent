import os
import chromadb
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# 1. Define the production-ready JSON schema using Pydantic
class DatabaseQueryPlan(BaseModel):
    selected_table: str = Field(description="The name of the database table selected for this query.")
    explanation: str = Field(description="The logical reasoning behind the SQL query construction.")
    sql_query: str = Field(description="The clean, executable SQL query string.")
    estimated_complexity: str = Field(description="Complexity level: Low, Medium, or High.")

print("🤖 Initializing Local AI Agent Components...")

# 2. Connect to the existing local ChromaDB instance
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
collection = chroma_client.get_collection(name="database_schemas")

# 3. Initialize local Llama 3.2 embeddings and LLM engine
embeddings = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(
    model="llama3.2",
    temperature=0.0,
    format="json"  # Forces the local model to return strictly valid JSON
)

# 4. Bind the structured output schema to our local LLM
structured_llm = llm.with_structured_output(DatabaseQueryPlan)

def run_data_agent(user_query: str):
    """
    Agentic Workflow: 
    1. Retrieves relevant database schema from local ChromaDB via vector search.
    2. Constructs a context-aware system prompt.
    3. Invokes the local LLM to generate a structured SQL plan.
    """
    print(f"\n🔍 Step 1: Querying Vector DB for context regarding: '{user_query}'")
    
    # Generate embedding vector for the user's natural language question
    query_vector = embeddings.embed_query(user_query)
    
    # Retrieve the single most relevant table schema from ChromaDB
    db_results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    retrieved_schema = db_results['documents'][0][0]
    associated_table = db_results['metadatas'][0][0]['table_name']
    
    print(f"🎯 Step 2: Found relevant table context -> [{associated_table}]")
    print(f"🧠 Step 3: Generating execution plan via Llama 3.2...")
    
    # Build the strict engineering prompt
    system_prompt = (
        f"You are an expert enterprise database architect. "
        f"Generate a safe, valid JSON database query plan based ONLY on this retrieved database schema:\n"
        f"{retrieved_schema}\n\n"
        f"Do not guess column names. Adhere strictly to the Pydantic JSON structure requested."
    )
    
    # Run the model
    agent_output = structured_llm.invoke([
        ("system", system_prompt),
        ("human", user_query)
    ])
    
    return agent_output

# --- Execute the Local Agent ---
if __name__ == "__main__":
    # Test Case 1: Inventory Question
    user_input_1 = "Alert me if any of our warehouse products have dropped below 15 items in stock."
    plan_1 = run_data_agent(user_input_1)
    
    print("\n================ AGENT OUTPUT 1 ================")
    print(f"Target Table : {plan_1.selected_table}")
    print(f"Reasoning    : {plan_1.explanation}")
    print(f"SQL Generated: {plan_1.sql_query}")
    print(f"Complexity   : {plan_1.estimated_complexity}")
    print("================================================\n")

    # Test Case 2: Sales Question
    user_input_2 = "What is the total revenue generated from customer purchases?"
    plan_2 = run_data_agent(user_input_2)
    
    print("================ AGENT OUTPUT 2 ================")
    print(f"Target Table : {plan_2.selected_table}")
    print(f"Reasoning    : {plan_2.explanation}")
    print(f"SQL Generated: {plan_2.sql_query}")
    print(f"Complexity   : {plan_2.estimated_complexity}")
    print("================================================")