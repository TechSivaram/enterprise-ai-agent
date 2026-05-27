import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Load environment variables from the .env file
load_dotenv()

# 1. Define the exact JSON schema we want the AI to return using Pydantic
class DatabaseQueryPlan(BaseModel):
    explanation: str = Field(description="The logical reasoning for why this query is written this way.")
    sql_query: str = Field(description="The clean, executable SQL query string.")
    estimated_complexity: str = Field(description="Low, Medium, or High depending on joins/subqueries.")

# 2. Initialize the high-capability chat model
# We set temperature=0.0 to ensure deterministic, precise technical outputs
llm = ChatOllama(
    model="llama3.2",
    temperature=0.0,
    format="json" # Tells the local model to strictly return valid JSON
)

# 3. Bind the structured output schema directly to the LLM engine
structured_llm = llm.with_structured_output(DatabaseQueryPlan)

def generate_query_plan(user_request: str, table_schema: str):
    """Uses the LLM to generate a safe, structured database execution plan."""
    system_prompt = f"You are an expert database architect. Convert the user request into valid SQL using this schema:\n{table_schema}"
    
    # Invoke the model with a structured conversation layout
    response = structured_llm.invoke([
        ("system", system_prompt),
        ("human", user_request)
    ])
    return response

# --- Test the Implementation ---
if __name__ == "__main__":
    # Mock database schema
    sample_schema = """
    Table: customer_orders
    Columns: order_id (INT), customer_name (VARCHAR), total_amount (DECIMAL), order_date (DATE)
    """
    
    request = "Show me the customer who spent the 2nd highest money in the year 2025."
    
    print("🤖 Processing request via LLM...")
    plan = generate_query_plan(request, sample_schema)
    
    print("\n--- Structured Output Received ---")
    print(f"Reasoning: {plan.explanation}")
    print(f"SQL Generated: {plan.sql_query}")
    print(f"Complexity: {plan.estimated_complexity}")