import argparse
import os
import re
import json
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def preprocess_text(text: str) -> str:
    """
    預處理文字：
      僅刪除指定的標點符號，例如中文逗號 (，) 與中文句號 (。) 等，
      保留其他符號（例如英文逗號、英文句點、小數點等）。
    """
    # 定義要刪除的字符集，可根據需要新增其他符號
    chars_to_remove = "，。、"
    # 使用正則表達式刪除指定字符
    cleaned_text = re.sub(f"[{chars_to_remove}]", "", text)
    # 移除多餘空白
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def assign_stock_labels(chunks: list[Document]) -> list[Document]:
    """
    根據每個 chunk 的 metadata 中的 source（例如檔案名稱），
    判斷屬於哪個個股報告，並在 metadata 中新增 "stock" 欄位。
    預設的股票列表包含：鴻海、東哥、緯穎、智崴和群聯。
    """
    stocks = ["鴻海", "東哥", "緯穎", "智崴", "群聯"]
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        assigned = False
        for stock in stocks:
            if stock in source:
                chunk.metadata["stock"] = stock
                assigned = True
                break
        if not assigned:
            chunk.metadata["stock"] = "未知"
    return chunks


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    chunks = assign_stock_labels(chunks)
    
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    #for chunk in chunks:
        # 將原始內容替換成整理後的結果
        #chunk.page_content = preprocess_text(chunk.page_content)
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    save_chunks_to_json(chunks_with_ids, json_file="data.json")

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        #db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def save_chunks_to_json(chunks: list[Document], json_file: str = "chunks.json"):
    """
    將切好的 chunks 存成 JSON 檔案，方便檢查每個 chunk 的內容與 metadata，
    並包含 chunk_id。
    """
    chunks_list = []
    for chunk in chunks:
        chunk_id = chunk.metadata.get("id", None)
        chunks_list.append({
            "chunk_id": chunk_id,
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        })
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(chunks_list, f, ensure_ascii=False, indent=4)
    print(f"Chunks saved to {json_file}")



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
