import argparse
import json
from lib.search_utils import GOLDEN_DATASET, load_movies
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    k = 60

    with open(GOLDEN_DATASET, "r") as f:
        dataset = json.load(f)
    
    test_cases = dataset["test_cases"]
    movies = load_movies()

    model = HybridSearch(movies)

    test_results = []
    for test_case in test_cases:
        query = test_case["query"]
        relevant_results = test_case["relevant_docs"]

        results = model.rrf_search(query, k, limit)

        relevant_retrieved = 0
        total_retreived = len(results)
        for movie in results:
            if movie["title"] in relevant_results:
                relevant_retrieved += 1
        
        precision = relevant_retrieved / total_retreived
        test_result = {
            "query": query,
            "precision": precision,
            "retrieved": [x["title"] for x in results],
            "relevant": [x for x in relevant_results]
        }
        test_results.append(test_result)

    for test in test_results:
        print(f"Query: {test["query"]}")
        print(f"Precision@{limit}: {test["precision"]:.4f}")
        print(f"Retrieved: {test['retrieved']}")
        print(f"Relevant: {test["relevant"]}")

if __name__ == "__main__":
    main()