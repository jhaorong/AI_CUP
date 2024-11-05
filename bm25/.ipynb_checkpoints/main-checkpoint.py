import os
import json
import jieba
import pdfplumber
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi

# === 1. 確保資料夾存在 ===
def ensure_dir_exists(path):
    """如果目錄不存在，則建立目錄"""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# === 2. 使用 ThreadPoolExecutor 加速 PDF 讀取 ===
def read_pdf(pdf_loc):
    """讀取單個 PDF 文件並返回其文字內容"""
    try:
        with pdfplumber.open(pdf_loc) as pdf:
            pdf_text = ''.join(page.extract_text() or '' for page in pdf.pages)
        return pdf_text
    except Exception as e:
        print(f"Error reading {pdf_loc}: {e}")
        return ''

def load_data(source_path):
    """載入 PDF 資料，並行處理以加快速度"""
    try:
        file_list = [f for f in os.listdir(source_path) if f.endswith('.pdf')]
        corpus_dict = {}

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(
                lambda f: (int(f.replace('.pdf', '')), read_pdf(os.path.join(source_path, f))),
                file_list), total=len(file_list)))

        corpus_dict = {key: value for key, value in results}
        return corpus_dict
    except FileNotFoundError:
        print(f"Error loading data: Directory not found -> {source_path}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return {}

# === 3. BM25 檢索 ===
def BM25_retrieve(qs, source, corpus_dict):
    """使用 BM25 演算法進行檢索"""
    filtered_corpus = [corpus_dict.get(int(f), '') for f in source]
    filtered_corpus = [doc for doc in filtered_corpus if doc.strip()]

    if not filtered_corpus:
        print(f"No valid documents found for query: {qs}")
        return None

    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(qs))

    top_n = bm25.get_top_n(tokenized_query, filtered_corpus, n=1)
    res = [key for key, value in corpus_dict.items() if value in top_n]
    return res[0] if res else None

# === 4. 主程式 ===
def main(question_path, source_path, output_path):
    """主程式：讀取問題、進行檢索並儲存結果"""

    # 確保輸出目錄存在
    ensure_dir_exists(output_path)

    # 讀取問題檔案
    try:
        with open(question_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Question file not found -> {question_path}")
        return

    # 載入資料
    corpus_dict_insurance = load_data(os.path.join(source_path, 'insurance'))
    corpus_dict_finance = load_data(os.path.join(source_path, 'finance'))

    # 讀取 FAQ JSON 檔案
    faq_path = os.path.join(source_path, 'faq', 'pid_map_content.json')
    try:
        with open(faq_path, 'r', encoding='utf-8') as f:
            key_to_source_dict = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        print(f"Error: FAQ file not found -> {faq_path}")
        key_to_source_dict = {}

    answer_dict = {"answers": []}

    # 根據每個問題進行檢索
    for q_dict in questions['questions']:
        category = q_dict['category']
        query = q_dict['query']
        source = q_dict['source']

        if category == 'finance':
            retrieved = BM25_retrieve(query, source, corpus_dict_finance)
        elif category == 'insurance':
            retrieved = BM25_retrieve(query, source, corpus_dict_insurance)
        elif category == 'faq':
            corpus_dict_faq = {k: str(v) for k, v in key_to_source_dict.items() if k in source}
            retrieved = BM25_retrieve(query, source, corpus_dict_faq)
        else:
            print(f"Error: Unknown category '{category}' in question {q_dict['qid']}")
            continue

        if retrieved is not None:
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

    # 儲存檢索結果
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_path}")

# === 5. 執行程式 ===
if __name__ == "__main__":
    # 設定路徑
    question_path = './dataset/preliminary/questions_example.json'
    source_path = './reference'
    output_path = './data/output/answers.json'

    # 檢查工作目錄是否正確
    print(f"Current Working Directory: {os.getcwd()}")

    # 執行主程式
    main(question_path, source_path, output_path)
