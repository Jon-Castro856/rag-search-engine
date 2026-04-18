import json
import os
from dotenv import load_dotenv
from google import genai
from typing import Any

MODEL_NAME = "gemma-3-27b-it"
DEFAULT_ALPHA = 0.5
DEFAULT_K = 60
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 500
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_OVERLAP = 1

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
EMBED_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDS = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA = os.path.join(CACHE_DIR, "chunk_metadata.json")
GOLDEN_DATASET = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
IMAGE = os.path.join(PROJECT_ROOT, "data", "paddington.jpg")


def load_golden_dataset() -> dict:
    with open(GOLDEN_DATASET, "r") as f:
        return json.load(f)
    
def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()

def load_llm() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    return client

def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

spell_check_prompt = """Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

rewrite_prompt = """Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

expand_prompt = """Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

rerank_prompt = """Rate how well this movie matches the search query.

Query: "{query}"
Movie: "{title}" - "{doc}"

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

batch_rank_prompt = """Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

eval_prompt = """Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{results}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

rag_prompt = """You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}"""

summarize_prompt = """Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

citations_prompt = """Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

question_prompt = """Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

other_question_prompt = prompt = """Answer the following question based on the provided documents.

Question: {question}

Documents:
{context}

General instructions:
- Answer directly and concisely
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Answer:"""

image_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""