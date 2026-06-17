from rag_pipeline import get_answer_with_sources

QUESTION = "What is the main finding of this paper?"

result = get_answer_with_sources(QUESTION)

print("ANSWER:")
print(result["answer"])
print()
print("SOURCES:")
for source in result["sources"]:
    print(f"  File: {source['file']} | Page: {source['page']}")
    print(f"  Snippet: {source['snippet']}")
    print()
