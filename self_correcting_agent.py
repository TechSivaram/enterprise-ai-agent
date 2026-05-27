import sqlite3
import chromadb
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# 1. Define the Pydantic structured output model
class SQLAgentResponse(BaseModel):
    explanation: str = Field(description="Reasoning behind the query adjustments.")
    sql_query: str = Field(description="The clean, executable SQL query string.")

print("🚀 Starting Self-Correcting Local AI Agent...")

# 2. Setup a temporary in-memory SQLite Database for execution testing
db_connection = sqlite3.connect(":memory:")
cursor = db_connection.cursor()

# Create the physical table matching our ChromaDB schema
cursor.execute("""
CREATE TABLE product_inventory (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    stock_quantity INTEGER,
    unit_price DECIMAL
);
""")
# Insert mock data
cursor.executemany("""
INSERT INTO product_inventory VALUES (?, ?, ?, ?);
""", [
    (101, "Industrial Solar Inverter", 8, 1200.00),
    (102, "Lithium-Ion Battery Pack", 25, 850.50),
    (103, "Monocrystalline Solar Panel", 4, 299.99)
])
db_connection.commit()

# 3. Connect to local ChromaDB and LLM components
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
collection = chroma_client.get_collection(name="database_schemas")
embeddings = OllamaEmbeddings(model="llama3.2")

llm = ChatOllama(model="llama3.2", temperature=0.0, format="json")
structured_llm = llm.with_structured_output(SQLAgentResponse)

def execute_and_correct_sql(user_query: str, max_attempts: int = 3):
    """
    Agentic loop that executes generated SQL queries, catches execution errors, 
    and requests the local LLM to self-correct based on error logs.
    """
    # Step 1: Retrieve context from ChromaDB
    query_vector = embeddings.embed_query(user_query)
    db_results = collection.query(query_embeddings=[query_vector], n_results=1)
    retrieved_schema = db_results['documents'][0][0]
    
    # Base system prompt instructing the model
    system_prompt = (
        f"You are an expert database engineer. Generate valid SQL matching this schema:\n"
        f"{retrieved_schema}\n"
        f"Respond strictly in the requested JSON format."
    )
    
    current_prompt = user_query
    conversation_history = [("system", system_prompt)]
    
    # Begin the autonomous self-correction loop
    for attempt in range(1, max_attempts + 1):
        print(f"\n🔄 [Attempt {attempt}/{max_attempts}] LLM processing query analysis...")
        
        conversation_history.append(("human", current_prompt))
        agent_output = structured_llm.invoke(conversation_history)
        
        generated_sql = agent_output.sql_query
        print(f"🤖 Generated SQL: {generated_sql}")
        
        try:
            # Attempt to execute the query against our SQLite instance
            cursor.execute(generated_sql)
            results = cursor.fetchall()
            print("✅ SQL Execution Successful!")
            return results, generated_sql
            
        except sqlite3.Error as database_error:
            print(f"❌ Database Execution Failed! Error: {database_error}")
            
            # If it fails, append the output and the exact database error back to context
            conversation_history.append(("ai", f"My previous query was: {generated_sql}"))
            
            # Update the prompt forcing the model to fix its specific error
            current_prompt = (
                f"The previous SQL query failed with this specific error message: '{database_error}'. "
                f"Analyze the mistake, fix the syntax or column names, and output a corrected SQL query."
            )
            
    print("\n🚨 Agent failed to resolve the database issue within maximum attempts.")
    return None, None

# --- Test the Self-Correction Capability ---
if __name__ == "__main__":
    # We pass a slightly trickier question to see the agent work
    user_request = "List our products out of stock, order them by stock level using a column called 'qty_left'."
    
    # Note: 'qty_left' does not exist in the schema (the column is stock_quantity). 
    # The agent must catch the error, map it correctly to stock_quantity, and re-run.
    
    data_results, successful_sql = execute_and_correct_sql(user_request)
    
    print("\n================ FINAL RESULTS ================")
    print(f"Executed SQL : {successful_sql}")
    print(f"Data Returned: {data_results}")
    print("===============================================")