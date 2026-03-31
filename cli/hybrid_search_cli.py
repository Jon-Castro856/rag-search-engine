import argparse

from lib.hybrid_search import normalize_score

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="normalize provided score values")
    normalize.add_argument("scores", nargs="+", type=float, help="numbers to be normalized")
    args = parser.parse_args()

    match args.command:
        case "normalize":
            res = normalize_score(args.scores)
            for n in res:
                print(f"* {n:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()