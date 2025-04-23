import lazyllm
from lazyllm import Document, Retriever

documents = Document("LazyLLM-RAG-Demo/dataset")
retriever = lazyllm.Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="bm25_chinese",
    topk=1,
    output_format="content",
    join='\n' + '='*200 + '\n'
)
retriever.start()
prompt = (
    'You will act as an AI question-answering assistant and complete a dialogue task.'
    'In this task, you need to provide your answers based on the given context and questions.'
)
llm = lazyllm.OnlineChatModule(source='glm').prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))
query = "脑机接口在医疗领域有哪些应用？\n"
retriever_res= retriever(query=query)
print("retriever_res:", '\n', '='*200, f"\n{retriever_res}\n", '='*200, '\n')
res = llm({"query": query, "context_str": retriever_res})
print(f"With RAG Answer:\n{res}")