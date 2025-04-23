import lazyllm

llm_prompt = "你是一只小猫, 每次回答完问题都要加上喵喵喵"
llm = lazyllm.OnlineChatModule(source='glm').prompt(llm_prompt)
res = llm({"query": "早上好",})
print(res)

llm_server = lazyllm.WebModule(llm, port=range(23468, 23470)).start().wait() 
