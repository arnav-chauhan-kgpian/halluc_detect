import logging
from pathlib import Path
from config import PipelineConfig
from data_loader import load_wildchat_queries

# Setup logging to see output
logging.basicConfig(level=logging.INFO)

def test_loading():
    # Use mock_data.jsonl which we know has 2 valid queries
    mock_path = Path("mock_data.jsonl")
    if not mock_path.exists():
        print("FAILURE: mock_data.jsonl not found")
        return

    cfg = PipelineConfig(data_path=mock_path, max_queries=10)
    queries = load_wildchat_queries(cfg)
    
    print(f"Loaded {len(queries)} queries")
    for i, q in enumerate(queries):
        print(f"Query {i+1}: ID={q['conversation_hash']}, Category={q['category']}")
        print(f"  Text: {q['query_text'][:50]}...")

    if len(queries) == 2:
        print("SUCCESS: Loaded all 2 queries from mock_data.jsonl")
    else:
        print(f"FAILURE: Expected 2 queries, but loaded {len(queries)}")

if __name__ == "__main__":
    test_loading()
