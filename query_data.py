import argparse
import sys
import time
import threading
import itertools
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
請完全根據以下個股投資報告用繁體中文回答問題

{context}

---

請完全根據上述投資報告的資料用繁體中文回答此問題: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text,stock)


def query_rag(query_text: str, stock: str) -> dict:
    """
    傳入 query_text（問題），
    執行以下步驟：
      1. 利用 Chroma 進行 similarity search，
      2. 用 prompt 組合上下文與問題，
      3. 使用 Ollama 的 mistral LLM 產生回答，
      4. 回傳回答與來源。
    """
    # 取得 embedding function 並初始化 DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 搜尋與問題相似的文本
    results = db.similarity_search_with_score(query_text, k=5,filter={"stock": stock})
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # 使用 prompt 模板填入上下文與問題
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    '''# 啟動等待動畫（spinner）
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    spinner_thread.start()
    '''
    # 呼叫 LLM 取得回答
    model = OllamaLLM(model="chatglm3")
    response_text = model.invoke(prompt)

    ''' # LLM 回答完成，停止等待動畫
    stop_event.set()
    spinner_thread.join()
    '''

    # 取得來源（假設文件 metadata 中的 id 為來源標識）
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    return {"response": response_text, "sources": sources}

# 定義等待動畫的函數
def spinner(stop_event):
    spinner_cycle = itertools.cycle(['-', '\\', '|', '/'])
    while not stop_event.is_set():
        sys.stdout.write("\rLLM 正在生成答案，請稍候... " + next(spinner_cycle))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\rLLM 生成完成！                \n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
