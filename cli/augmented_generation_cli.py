import argparse
from lib.search_utils import load_movies, load_llm, MODEL_NAME, rag_prompt, summarize_prompt
from lib.hybrid_search import HybridSearch

def main():
    parser = init_parser()

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rag_search(query)

            format_results = []
            for res in results:
                formatting = [res["title"], res["document"]]
                format_results.append(chr(10).join(formatting))

            llm_response = query_llm(rag_prompt.format(query=query, docs=chr(10).join(format_results)))

            print(f"Showing search results for {query}")
            for res in results:
                print(f"- {res["title"]}")

            print("\n")
            print("RAG response:")
            print(llm_response if llm_response else "No response")

        case "summarize":
            query = args.query
            results = rag_search(query)

            format_results = []
            for res in results:
                formatting = [res["title"], res["document"]]
                format_results.append(chr(10).join(formatting))

            llm_response = query_llm(summarize_prompt.format(query=query, results=chr(10).join(format_results)))

            print(f"Showing search results for {query}")
            for res in results:
                print(f"- {res["title"]}")

            print("\n")
            print("LLM Summary:")
            print(llm_response if llm_response else "No response")
        case _:
            parser.print_help()

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize = subparsers.add_parser("summarize", help="Have an LLM provide an indepth explanation of your search query and resukts")
    summarize.add_argument("query", type=str, help="query to search")
    summarize.add_argument("--limit", default=5, type=int, help="maximum number of results to show")
    
    return parser

def rag_search(query: str):
    movies = load_movies()
    model = HybridSearch(movies)

    results = model.rrf_search(query, limit=5)

    return results

def query_llm(prompt: str) -> str:
    client = load_llm()
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

    return response.text

if __name__ == "__main__":
    main()