import argparse
import sys
import time
import re
import threading
import itertools
import json
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

def spinner(stop_event):
    """顯示等待動畫"""
    spinner_cycle = itertools.cycle(['-', '\\', '|', '/'])
    while not stop_event.is_set():
        sys.stdout.write("\rLLM 正在生成答案，請稍候... " + next(spinner_cycle))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\rLLM 生成完成！                \n")
    sys.stdout.flush()

def load_question_set(file_path: str) -> list:
    """
    如果有提供 JSON 檔，格式預期為列表，
    每個元素為字典，包含 "question" 與 "expected_answer" 兩個欄位。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    return questions

def preprocess_text(text: str) -> str:
    """
    預處理文字：
      僅刪除指定的標點符號，例如中文逗號 (，) 與中文句號 (。) 等，
      保留其他符號（例如英文逗號、英文句點、小數點等）。
    """
    # 定義要刪除的字符集，可根據需要新增其他符號
    chars_to_remove = "，。、?"
    # 使用正則表達式刪除指定字符
    cleaned_text = re.sub(f"[{chars_to_remove}]", "", text)
    # 移除多餘空白
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_set", type=str, default='./query_temp.json',
                        help="問題集 JSON 檔案路徑（每個元素需包含 'question' 與 'expected_answer'）")
    args = parser.parse_args()

     # 如果 args 中沒有 output 屬性，給予預設值
    if not hasattr(args, "output"):
        args.output = "output_4.json"

    # 如果使用者有提供問題集檔案，從檔案讀入；否則使用內建的預設問題集
    if args.question_set:
        questions = load_question_set(args.question_set)

    evaluation_results = []
    # 對每個問題依序執行 RAG 查詢
    for entry in questions:
        question_text = entry["question"]
        stock = entry["stock"]
        print(f"\n正在處理問題：{question_text} （個股：{stock}）")
        result = query_rag(question_text, stock)
        print(f"答案：{result['response']}")
        evaluation_entry = {
            "question": question_text,
            "expected_answer": entry["expected_answer"],
            "stock": stock,
            "response": result["response"],
            "sources": result["sources"]
        }
        evaluation_results.append(evaluation_entry)

    # 輸出 JSON 格式的評估結果
    output_json = json.dumps(evaluation_results, ensure_ascii=False, indent=4)
    print("\n最終評估結果：")
    print(output_json)

    # 將評估結果存成 JSON 檔
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"\n評估結果已存檔至：{args.output}")


if __name__ == "__main__":
    main()
