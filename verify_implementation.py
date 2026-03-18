import sys
import os
sys.path.append(r"d:\halluc_detect")

from utils.similarity_analysis import run_similarity_analysis
from data_loader import load_wildchat_queries
from config import PipelineConfig

def test_data_loader():
    cfg = PipelineConfig(data_path="d:\\halluc_detect\\mock_data.jsonl", max_queries=10)
    queries = load_wildchat_queries(cfg)
    print(f"Loaded {len(queries)} queries.")
    for q in queries:
        print(f"Hash: {q['conversation_hash']}, Category: {q['category']}")
        print(f"Query snippet: {q['query_text'][:50]}...")

def test_similarity():
    gen_text = "```cpp\nint add(int a, int b) {\n    return a + b;\n}\n```"
    src_text = "```python\ndef add(a, b):\n    return a + b\n```"
    metrics = run_similarity_analysis(gen_text, src_text)
    print(f"Similarity Metrics: {metrics}")

if __name__ == "__main__":
    print("--- Testing Data Loader ---")
    test_data_loader()
    print("\n--- Testing Similarity Analysis ---")
    test_similarity()
