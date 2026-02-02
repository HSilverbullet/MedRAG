import os
import numpy as np
import torch
from openai import OpenAI
from langchain_community.vectorstores import FAISS

# 只保留一个 rag 模块，避免覆盖
from .rag import (
    init_embeddings,
    load_and_process_pubmed,
    split_documents,
    build_vector_db,
    build_bm25_index,
    load_bm25_index,
    bm25_search,
    init_bge_reranker,
    rerank_with_bge,
)

from .translate import BaiduGeneralTranslator


class OpenAIAPI:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
        )

        self.system_prompt = """你是专业的医学智能问答助手，需严格按以下规则回答：
        1. 仅回答医学、健康相关问题；其他问题直接拒绝，回复“抱歉，我仅能解答医学领域相关问题，请提问医学相关内容”。
        2. 若提供的“参考文献”为“（无相关文献）”，直接基于医学常识回答，且不要编造或虚构文献；此时不要输出参考文献列表。
        3. 若提供了相关文献，优先基于文献内容回答，语言通俗易懂；正文中不要输出“参考文献”或引用列表（由系统在回答末尾统一追加）。
        4. 所有回答必须使用中文。
        """

        # 路径配置
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.current_dir, "../pubmed_db")
        self.json_path = os.path.join(self.current_dir, "../data/pubmed_data.json")

        # 新增：BM25 索引路径（与 faiss 同目录即可）
        self.bm25_index_name = "bm25.pkl"

        # 召回/重排参数（你可以按效果调）
        self.faiss_recall_k = 20
        self.bm25_recall_k = 30
        self.final_top_k = 4  # 最终给 LLM 的文献数

        # 初始化翻译器
        self.translator = self.init_translator()

        # 初始化向量库 + BM25 + Reranker
        self.init_retrieval_components()

    def init_translator(self):
        try:
            return BaiduGeneralTranslator()
        except Exception as e:
            print(f"百度翻译器初始化失败：{e}，将使用原始文本检索")
            return None

    def translate_medical_query(self, query: str) -> str:
        if not self.translator:
            return query
        return self.translator.translate(query)

    def init_retrieval_components(self):
        # embeddings
        device = "cuda" if os.environ.get("USE_CUDA") else "cpu"
        self.embeddings = init_embeddings(device=device)

        # 1) FAISS
        faiss_ok = os.path.exists(f"{self.db_path}/index.faiss") and os.path.exists(f"{self.db_path}/index.pkl")

        # 2) BM25
        bm25_ok = os.path.exists(os.path.join(self.db_path, self.bm25_index_name))

        # 先确保 chunks 存在（懒加载）
        chunks = None

        # ---------- FAISS ----------
        if faiss_ok:
            self.db = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("FAISS 向量库加载完成")
        else:
            print("未找到 FAISS 索引，开始重建 FAISS...")
            if chunks is None:
                documents = load_and_process_pubmed(self.json_path)
                chunks = split_documents(documents)
            self.db = build_vector_db(chunks, self.embeddings, self.db_path)

        # ---------- BM25 ----------
        if bm25_ok:
            self.bm25, self.bm25_chunks = load_bm25_index(
                self.db_path,
                self.bm25_index_name
            )
            print("BM25 索引加载完成")
        else:
            print("未找到 BM25 索引，开始重建 BM25...")
            if chunks is None:
                documents = load_and_process_pubmed(self.json_path)
                chunks = split_documents(documents)
            self.bm25, self.bm25_chunks = build_bm25_index(
                chunks=chunks,
                save_path=self.db_path,
                index_name=self.bm25_index_name
            )

        # 3) BGE Reranker
        rerank_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = init_bge_reranker(model_name=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base"),
                                          device=rerank_device)

        print("检索组件初始化完成（FAISS + BM25 + BGE Reranker）")

    def _merge_dedup_docs(self, docs_a, docs_b):
        """按 title+source 去重合并，保留更完整的 page_content（简单策略）"""
        uniq = {}
        for d in (docs_a + docs_b):
            title = d.metadata.get("title", "")
            source = d.metadata.get("source", "")
            key = f"{title}__{source}"
            if key not in uniq:
                uniq[key] = d
            else:
                # 选更长内容作为代表
                if len(d.page_content) > len(uniq[key].page_content):
                    uniq[key] = d
        return list(uniq.values())

    def retrieve_relevant_docs(self, query: str):
        # 把中文提问翻译为英文
        translated_query = self.translate_medical_query(query)

        # 1) BM25 关键词召回（更容易命中专有名词、药名、基因名等）
        bm25_docs = bm25_search(self.bm25, self.bm25_chunks, translated_query, top_k=self.bm25_recall_k)

        # 2) FAISS 向量召回（语义召回）
        faiss_docs = self.db.similarity_search(translated_query, k=self.faiss_recall_k)

        # 3) 合并去重
        candidates = self._merge_dedup_docs(bm25_docs, faiss_docs)
        if not candidates:
            return []

        # 4) BGE 重排（对 query-doc 相关性做更强判别）
        reranked = rerank_with_bge(self.reranker, translated_query, candidates, top_k=self.final_top_k)

        return reranked

    def get_response(self, chat_history):
        try:
            # 1. 输入校验
            user_query = chat_history[-1]["content"].strip()
            if not user_query:
                return "请输入有效的医学问题。"

            # 2. 检索文献
            relevant_docs = self.retrieve_relevant_docs(user_query)

            # 3. 统一构造上下文与来源
            context_list = []
            source_list = []

            for i, doc in enumerate(relevant_docs):
                title = doc.metadata.get('title', '未知文献')
                # 改进点：不再暴力截断，保留检索到的完整语义块，或按需适度增加上限
                content = doc.page_content.strip()

                # 构造给 LLM 的上下文 (带序号增强关联性)
                context_list.append(f"【文献{i + 1}】标题：{title}\n内容：{content}")
                # 构造展示给用户的来源列表
                source_list.append(f"- {title}")

            # 4. 判定是否有参考资料
            if context_list:
                context_str = "\n\n".join(context_list)
                sources_footer = "\n".join(source_list)
            else:
                context_str = "（无相关文献）"
                sources_footer = "未检索到相关医学文献"

            # 5. 构建 Prompt 并调用模型
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"用户问题：{user_query}\n参考文献：\n{context_str}"}
            ]

            response = self.client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )

            answer = response.choices[0].message.content

            # 6. 格式化输出
            if context_list:
                final_response = f"{answer}\n\n参考文献：\n{sources_footer}"
            else:
                final_response = answer

            return final_response

        except Exception as e:
            # 打印详细日志以便排查，给用户返回友好提示
            print(f"Error in get_response: {str(e)}")
            return "抱歉，系统处理您的请求时出现了医学知识库连接异常，请稍后再试。"
