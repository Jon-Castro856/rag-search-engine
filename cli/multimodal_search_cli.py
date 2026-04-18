import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal CLI search")
    subparsers = parser.add_subparsers(dest="command", help="")
    verify = subparsers.add_parser("verify_image_embedding", help="create embedding of image")
    verify.add_argument("img_path", type=str, help="image path")
    image_search = subparsers.add_parser("image_search", help="search using an image")
    image_search.add_argument("img_path", type=str, help="path of image")

    args = parser.parse_args()
    match args.command:
        case "image_search":
            results = image_search_command(args.img_path)

            for i, res in enumerate(results):
                print(f"{i+1}. {res["title"]} (similarity: {res["score"]:.3f})")
                print(f"{res["description"][:100]}...")
if __name__ == "__main__":
    main()