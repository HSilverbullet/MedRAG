# MedRAG: 医学文献智能检索与问答系统
数据自主抓取、混合索引构建 与 二阶段精排 RAG
这是一个集成了 **数据自主抓取**、**混合索引构建** 与 **二阶段精排** 的 **RAG** 技术医学文献处理平台。系统能够从 PubMed 自动获取医学前沿文献，并通过语义与关键词双驱动的检索方式，配合深度学习重排序模型，为用户提供极高准确度的医学知识问答。

## 项目结构

```text
├── data/
│   ├── download_pubmed.py    # PubMed 文献抓取脚本 
│   └── pubmed_data.json      # 抓取并结构化后的文献数据库
├── MedRAG/
│   ├── main.py               # 系统集成调度中心
│   ├── rag.py                # RAG 检索与生成核心逻辑
│   └── translate.py          # 问题翻译模块
├── pubmed_db/
│   ├── bm25.pkl              # BM25 关键词检索索引文件
│   ├── index.faiss           # FAISS 向量索引文件 (存储 Embeddings)
│   └── index.pkl             # FAISS 对应的元数据映射
└── app.py                    # Streamlit 交互式 Web 应用界面

```

## 运行指南

### 1. 环境配置

首先克隆项目并安装必要的依赖：

```bash
pip install streamlit pandas faiss-cpu rank_bm25 openai langchain

```

### 2. 数据采集与结构化

运行以下命令，系统将根据预设关键词从 PubMed 抓取核心摘要并转换为 `JSON` 格式：

```bash
python data/download_pubmed.py

```

### 3. 启动应用

直接启动 Web 界面即可开始交互：

```bash
streamlit run app.py

```

## 技术架构

本系统采用双路检索架构，确保医学咨询的准确性：

1. **第一阶段：混合检索 (Hybrid Retrieval)**

* **向量检索 (Dense)**：利用 FAISS 捕获语义信息，解决“同义词”和“隐性相关”的匹配问题。

* **传统检索 (Sparse)**：利用 BM25 确保对药物名、病理术语等硬核关键信息的精确命中。

2. **第二阶段：二阶段精排 (Reranking)**

* **BGE Reranker**：引入`bge-reranker-base`模型，对粗排召回的候选片段进行二次打分。

* **核心证据筛选**：通过深度语义比对，剔除噪音，仅将相关性评分最高的片段输入 LLM。

3. **第三阶段：生成与溯源 (Generation & Attribution)**

* 基于精排后的上下文生成回答，并在界面显示对应的文献来源，解决 AI 幻觉问题。

## 核心功能

* **自主抓取**：无需手动下载，一键同步 PubMed 最新研究。
* **极致精准**：通过 **BGE Reranker** 进行精排，在医学这种严谨场景下，确保证据的绝对相关。
* **多维检索**：结合 `BM25` 的“硬匹配”与 `FAISS` 的“软语义”，覆盖各类搜索需求。
* **医疗友好**：内置翻译模块，支持中文提问、英文检索、中文回答的跨语言知识获取。
* **可视化界面**：基于 Streamlit 打造，提供文献预览、来源溯源及 AI 深度对话。
