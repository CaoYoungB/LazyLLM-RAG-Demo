from lazyllm import Document, Retriever

doc = Document("LazyLLM-RAG-Demo/dataset")
seperator = '\n' + '='*200 + '\n'
retriever = Retriever(
    doc,
    group_name='CoarseChunk',
    similarity="bm25_chinese",
    topk=1,
    output_format="content",
    join=seperator
)
res = retriever("脑机接口在医疗领域有哪些应用？")
print(res)