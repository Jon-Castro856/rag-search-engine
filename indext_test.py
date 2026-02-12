from cli.inverted_index import MyInvertedIndex
from solution_code.cli.lib.keyword_search import InvertedIndex

my_index = MyInvertedIndex()
sol_index = InvertedIndex()

my_index.build()
sol_index.build()

print("length comparison")
print(f"---index---")
print(f"Mine: {len(my_index.index)}")
print(f"Solution: {len(sol_index.index)}")

print(f"---docmap---")
print(f"Mine: {len(my_index.docmap)}")
print(f"Solution: {len(sol_index.docmap)}")

print(f"---term frequency---")
print(f"Mine: {len(my_index.term_frequencies)}")
print(f"Solution: {len(sol_index.term_frequencies)}")

print(f"---doc lengths---")
print(f"Mine: {len(my_index.doc_lengths)}")
print(f"Solution: {len(sol_index.doc_lengths)}")