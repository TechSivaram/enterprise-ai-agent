import chromadb
from langchain_ollama import OllamaEmbeddings

class LocalSemanticCache:
    def __init__(self, threshold: float = 0.15):
        """
        Initializes a local semantic cache using ChromaDB.
        Threshold defines the maximum allowable distance for a match (lower = stricter).
        """
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
        
        # Create or fetch a collection dedicated to caching responses
        self.collection = self.chroma_client.get_or_create_collection(name="query_cache")
        self.threshold = threshold

    def lookup(self, user_query: str):
        """
        Performs a semantic lookup to check if a similar query exists in cache.
        """
        if self.collection.count() == 0:
            return None
            
        query_vector = self.embeddings.embed_query(user_query)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=1
        )
        
        # Check if the closest match falls within our semantic distance threshold
        if results and results['distances'][0]:
            distance = results['distances'][0][0]
            if distance <= self.threshold:
                print(f"🎯 Cache Hit! Semantic distance: {distance:.4f}")
                return results['documents'][0][0]  # Returns the cached SQL string
                
        print("🧊 Cache Miss. Processing query via AI Agent pipeline...")
        return None

    def update(self, user_query: str, generated_sql: str):
        """
        Saves a newly generated query and its SQL translation into the cache.
        """
        query_vector = self.embeddings.embed_query(user_query)
        # Use a simplified hash or clean string as the ID
        cache_id = f"cache_{hash(user_query)}"
        
        self.collection.add(
            ids=[cache_id],
            embeddings=[query_vector],
            documents=[generated_sql]
        )
        print("💾 Successfully cached query optimization parameters.")

# --- Verification Block ---
if __name__ == "__main__":
    cache = LocalSemanticCache()
    
    # Simulate first execution
    sample_query_1 = "List all orders from yesterday."
    sample_sql = "SELECT * FROM customer_orders WHERE order_date = '2026-05-26';"
    
    print("--- Execution 1 ---")
    hit = cache.lookup(sample_query_1)
    if not hit:
        # Code updates the cache after agent generation completes
        cache.update(sample_query_1, sample_sql)
        
    print("\n--- Execution 2 (Identical Request) ---")
    cached_result = cache.lookup(sample_query_1)
    print(f"Result: {cached_result}")
    
    print("\n--- Execution 3 (Semantically Similar Request) ---")
    # Notice the phrasing change; standard string matching fails here, but vector lookup passes
    similar_query = "Give me the order history for yesterday."
    cached_result_2 = cache.lookup(similar_query)
    print(f"Result: {cached_result_2}")