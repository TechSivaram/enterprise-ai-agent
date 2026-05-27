import chromadb
from langchain_community.embeddings import OllamaEmbeddings

print("📦 Initializing Local ChromaDB and Vector Embeddings...")

# 1. Initialize local Llama 3.2 embeddings model via Ollama
# This converts our textual schemas into dense numerical vectors
embeddings = OllamaEmbeddings(model="llama3.2")

# 2. Configure a persistent local ChromaDB client
# This creates a physical folder named './chroma_db_storage' in your project root
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")

# 3. Create or fetch a collection dedicated to storing database metadata
collection = chroma_client.get_or_create_collection(name="database_schemas")

# 4. Define enterprise database schemas (Mock Metadata Catalog)
schemas = [
    {
        "id": "schema_orders",
        "text": (
            "Table: customer_orders. "
            "Columns: order_id (INT, Primary Key), customer_name (VARCHAR), total_amount (DECIMAL), order_date (DATE). "
            "Use this table for sales, revenue, and customer purchase histories."
        ),
        "metadata": {"table_name": "customer_orders", "type": "sales"}
    },
    {
        "id": "schema_inventory",
        "text": (
            "Table: product_inventory. "
            "Columns: product_id (INT, Primary Key), product_name (VARCHAR), stock_quantity (INT), unit_price (DECIMAL). "
            "Use this table for stock levels, warehouse data, and product prices."
        ),
        "metadata": {"table_name": "product_inventory", "type": "inventory"}
    }
]

print("\n🔄 Vectorizing and injecting schemas into local ChromaDB storage...")

# 5. Process and insert each schema into the local vector database
for schema in schemas:
    # Compute the text vector embeddings locally
    vector_representation = embeddings.embed_query(schema["text"])
    
    # Add record to the collection
    collection.add(
        ids=[schema["id"]],
        embeddings=[vector_representation],
        documents=[schema["text"]],
        metadatas=[schema["metadata"]]
    )

print("✅ Local ChromaDB Vector Store populated successfully!")

# 6. Sanity Check: Run a local vector search test
print("\n🔍 Testing Vector Search: 'How many items are left in stock?'")
query_vector = embeddings.embed_query("How many items are left in stock?")

search_results = collection.query(
    query_embeddings=[query_vector],
    n_results=1  # Retrieve the single closest matching schema
)

print("\n🎯 Closest Database Schema Context Found:")
print(search_results['documents'][0][0])