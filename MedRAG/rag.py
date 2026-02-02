from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import re
import os
import json
import pickle
from typing import List, Tuple
from rank_bm25 import BM25Okapi


try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None


def metadata_func(record: dict, metadata: dict) -> dict:
    pub_date = record.get("pub_date", {})
    metadata["title"] = record.get("article_title", "")
    metadata["year"] = pub_date.get("year", "")
    metadata["month"] = pub_date.get("month", "")
    metadata["day"] = pub_date.get("day", "")
    # 可选：如果你的 record 里有 source 字段，否则你后面 doc.metadata['source'] 会 KeyError
    metadata["source"] = record.get("source", "pubmed")
    return metadata


def clean_text(text: str) -> str:
    if not text:
        return ""
    # 1. 去除 HTML 标签
    text = re.sub(r"<.*?>", "", text)
    # 2. 改进正则：保留字母、数字、基础标点，以及医学关键符号（如 % / + -）
    # 之前的正则 [^\w\s.,!?;:-] 会剔除掉 % 和 / (如 5mg/kg)
    text = re.sub(r"[^\w\s.,!?;:%\-\+\/]", "", text)
    # 3. 合并多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_process_pubmed(json_path: str) -> List[Document]:
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".[]",
        content_key="article_abstract",
        metadata_func=metadata_func
    )
    raw_documents = loader.load()
    print(f"原始数据加载完成，共 {len(raw_documents)} 篇文献")

    cleaned_documents = []
    for doc in raw_documents:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:
            cleaned_documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))

    print(f"文本清洗完成，有效文献数量：{len(cleaned_documents)}")
    return cleaned_documents


def split_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 60) -> List[Document]:
    """
    使用递归字符切分器。
    它会按照 [段落 > 换行 > 句子 > 空格] 的优先级切分，最大程度保护语义。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True  # 方便后续定位
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文本分块完成：使用递归切分，产生 {len(chunks)} 个片段")
    return chunks


def init_embeddings(model_name: str = "all-mpnet-base-v2", device: str = "cuda") -> HuggingFaceEmbeddings:
    # 嵌入模型的主要作用是将文本转化为向量表示，以便计算文本间的相似度，帮助模型从外部知识库中检索到相关信息，进而增强生成模型的答案质量。
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False}
    )
    print(f"嵌入模型初始化完成：{model_name}（运行设备：{device}）")
    return embeddings


def build_vector_db(chunks: List[Document], embeddings: HuggingFaceEmbeddings, save_path: str) -> FAISS:
    # 构建faiss向量数据库
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)
    print(f"向量数据库构建完成，已保存至 {save_path}")
    return db


# -------------------- BM25 --------------------

def _simple_tokenize(text: str) -> List[str]:
    # 你数据是英文 PubMed 摘要为主，这样分词足够；若你存的是中文，请换成 jieba 等
    # 简单的英文文本分词器，它将输入的文本转为小写并使用正则表达式提取其中的字母和数字部分
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def build_bm25_index(
        chunks: List[Document],
        save_path: str,
        index_name: str = "bm25.pkl"
) -> Tuple[BM25Okapi, List[Document]]:
    tokenized_corpus = [_simple_tokenize(d.page_content) for d in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, index_name), "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"BM25 索引构建完成，已保存至 {os.path.join(save_path, index_name)}")
    return bm25, chunks


def load_bm25_index(save_path: str, index_name: str = "bm25.pkl") -> Tuple[BM25Okapi, List[Document]]:
    p = os.path.join(save_path, index_name)
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["chunks"]


def bm25_search(
        bm25: BM25Okapi,
        bm25_chunks: List[Document],
        query: str,
        top_k: int = 10
) -> List[Document]:
    q_tokens = _simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    # 取 top_k 的索引
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [bm25_chunks[i] for i in top_idx]


# -------------------- BGE Reranker --------------------

def init_bge_reranker(model_name: str = "BAAI/bge-reranker-base", device: str = "cuda"):
    if FlagReranker is None:
        raise ImportError("未安装 FlagEmbedding，请先 pip install FlagEmbedding")
    reranker = FlagReranker(model_name, use_fp16=(device.startswith("cuda")))
    print(f"BGE Reranker 初始化完成：{model_name}（fp16={device.startswith('cuda')}）")
    return reranker


def rerank_with_bge(
        reranker,
        query: str,
        docs: List[Document],
        top_k: int = 5
) -> List[Document]:
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.compute_score(pairs)  # List[float]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked[:top_k]]
