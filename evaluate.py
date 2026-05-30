import time
from your_rag_file import qa_chain  # import however you built your chain

# ── Test Queries ──────────────────────────────
test_queries = [
    "What is the main topic of this document?",
    "Summarize the key points",
    "What are the important details?"
]

response_times = []
chunks_list = []

for query in test_queries:
    start = time.time()
    result = qa_chain({"query": query})
    end = time.time()
    
    elapsed = round(end - start, 2)
    chunks = len(result['source_documents'])
    
    response_times.append(elapsed)
    chunks_list.append(chunks)
    
    print(f"Query: {query}")
    print(f"Response Time: {elapsed}s")
    print(f"Chunks Retrieved: {chunks}")
    print(f"Answer: {result['result'][:100]}...")
    print("─" * 50)

# ── Summary ───────────────────────────────────
print("\n=== PERFORMANCE SUMMARY ===")
print(f"Avg Response Time : {round(sum(response_times)/len(response_times), 2)}s")
print(f"Avg Chunks Retrieved : {round(sum(chunks_list)/len(chunks_list), 1)}")
