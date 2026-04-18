import argparse
import mimetypes
from google.genai import types
from lib.search_utils import load_llm, image_prompt, MODEL_NAME

def main() -> None:
    parser = init_parser()
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        image = f.read()

    client = load_llm()

    parts=[
    image_prompt, types.Part.from_bytes(data=image, mime_type=mime), args.query.strip()]
    response = client.models.generate_content(model=MODEL_NAME, contents=parts )
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")




def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image search CLI")
    parser.add_argument("--image", type=str, help="path to image file")
    parser.add_argument("--query", type=str, help="query for searching")

    return parser

if __name__ == "__main__":
    main()