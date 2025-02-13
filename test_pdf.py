from langchain_community.document_loaders import PyPDFDirectoryLoader

# 指定存放 PDF 檔案的資料夾路徑
pdf_directory = "./data"

# 建立 loader 物件
loader = PyPDFDirectoryLoader(pdf_directory)

# 載入所有 PDF 文件
documents = loader.load()

# 檢視每個文件的內容與 metadata
for i, doc in enumerate(documents):
    print(f"Document {i+1}")
    print("Metadata:", doc.metadata)
    # 為了避免一次印出過多文字，這裡只顯示前 500 個字元
    print("Content Preview:", doc.page_content[:500])
    print("-" * 50)
