from rag_pipeline import get_answer_with_sources

questions = [
    "What deep learning models were used for Alzheimer's disease classification?",
    "What is the accuracy achieved in MRI classification?",
    "What dataset was used in this study?"
]

for q in questions:
    print(f"\nQ: {q}")
    result = get_answer_with_sources(q)
    print(f"A: {result['answer']}")
    print("Sources:")
    for s in result['sources'][:2]:  # show top 2 sources only
        print(f"  → {s['file']} | Page {s['page']}")
    print("-" * 60)