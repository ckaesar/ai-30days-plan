#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Qwen API的RAG系统完整实现
包含文档加载、向量化、检索、增强生成全流程
"""
import os
import json
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

load_dotenv()

class QwenRAGSystem:
    """基于Qwen的RAG系统"""
    
    def __init__(self, qwen_api_key: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """初始化RAG系统组件"""
        self.qwen_api_key = qwen_api_key
        self.base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.headers = {
            "Authorization": f"Bearer {qwen_api_key}",
            "Content-Type": "application/json"
        }
        
        # 初始化嵌入模型
        self.embedding_model = None
        if LANGCHAIN_AVAILABLE:
            print(f"加载嵌入模型: {embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name
            )
        
        # 初始化向量数据库
        self.vectorstore = None
        self.retriever = None
        self._chunks: List[str] = []
        self._df: Dict[str, int] = {}
        self._N: int = 0
        self._doc_vectors: List[Dict[str, float]] = None
    
    def load_and_chunk_documents(self, file_path: str) -> List[str]:
        """加载文档并进行智能分块"""
        # 简化示例：从文本文件加载
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # 递归分块：按段落→句子→字符逐级切分
        if LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]
            )
            chunks = text_splitter.split_text(full_text)
        else:
            chunks = []
            chunk_size = 500
            chunk_overlap = 50
            i = 0
            while i < len(full_text):
                end = min(len(full_text), i + chunk_size)
                chunks.append(full_text[i:end])
                if end >= len(full_text):
                    break
                i = max(0, end - chunk_overlap)
        print(f"文档分块完成：共{len(chunks)}个片段")
        self._chunks = chunks
        return chunks
    
    def build_vector_index(self, chunks: List[str], persist_dir: str = "./chroma_db"):
        """构建向量索引并持久化"""
        # 使用Chroma向量数据库
        if LANGCHAIN_AVAILABLE:
            self.vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=self.embedding_model,
                persist_directory=persist_dir
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            print(f"向量索引构建完成，持久化到: {persist_dir}")
            return
        self._chunks = chunks
        self._N = len(chunks)
        df_map = defaultdict(int)
        doc_tokens = []
        for c in chunks:
            toks = self._tokenize(c)
            doc_tokens.append(toks)
            for term in set(toks):
                df_map[term] += 1
        self._df = dict(df_map)
        self._doc_vectors = [self._tfidf(toks) for toks in doc_tokens]
        print("向量索引构建完成（本地TF-IDF回退实现）")

    def _tokenize(self, text: str) -> List[str]:
        seps = " \t\r\n.,;:!?'\"()[]{}<>，。！？；、：（）【】《》“”‘’|-_/\\"
        buf = []
        for ch in text.lower():
            buf.append(" " if ch in seps else ch)
        raw = "".join(buf).split()
        tokens: List[str] = []
        for w in raw:
            if any("\u4e00" <= c <= "\u9fff" for c in w):
                tokens.extend(list(w))
            else:
                tokens.append(w)
        return [t for t in tokens if t]

    def _tfidf(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec: Dict[str, float] = {}
        for term, f in tf.items():
            df = self._df.get(term, 0)
            idf = math.log((self._N + 1) / (df + 1)) + 1.0
            vec[term] = (f / len(tokens)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for k in list(vec.keys()):
            vec[k] /= norm
        return vec
    
    def retrieve_relevant_context(self, query: str) -> str:
        """检索与查询相关的上下文"""
        if LANGCHAIN_AVAILABLE:
            if not self.retriever:
                raise ValueError("请先构建向量索引")
            docs = self.retriever.get_relevant_documents(query)
            contexts = [doc.page_content for doc in docs]
        else:
            if not self._doc_vectors:
                raise ValueError("请先构建向量索引")
            q_vec = self._tfidf(self._tokenize(query))
            scores: List[tuple[float, int]] = []
            for idx, d_vec in enumerate(self._doc_vectors):
                dot = 0.0
                for term, qv in q_vec.items():
                    dv = d_vec.get(term)
                    if dv:
                        dot += qv * dv
                scores.append((dot, idx))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [self._chunks[i] for _, i in scores[:3]]
            contexts = top
        
        # 合并上下文，限制长度
        combined_context = "\n\n".join(contexts[:3])
        if len(combined_context) > 2000:
            combined_context = combined_context[:2000] + "..."
        
        print(f"检索到{len(contexts)}个相关片段")
        return combined_context
    
    def generate_enhanced_answer(self, query: str, context: str) -> str:
        """基于检索上下文生成增强回答"""
        # 构建增强提示
        enhanced_prompt = f"""
你是一个专业的AI助手，请基于以下检索到的信息回答问题：

【检索到的相关内容】
{context}

【用户问题】
{query}

要求：
1. 严格基于检索到的信息回答，不要编造
2. 如果信息不足，明确说明"根据现有信息无法回答"
3. 回答简洁准确，必要时标注信息来源
"""
        
        # 调用Qwen API
        payload = {
            "model": os.getenv("LLM_MODEL", "qwen-max"),
            "messages": [{"role": "user", "content": enhanced_prompt}],
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500"))
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            return answer
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            return "无法生成回答，请检查API连接"
    
    def query_rag_system(self, query: str) -> Dict[str, Any]:
        """完整的RAG查询流程"""
        print(f"处理查询: {query}")
        
        # 1. 检索相关上下文
        context = self.retrieve_relevant_context(query)
        
        # 2. 生成增强回答
        answer = self.generate_enhanced_answer(query, context)
        
        return {
            "query": query,
            "context": context,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }

# 使用示例
if __name__ == "__main__":
    from datetime import datetime
    
    # 初始化RAG系统（需要真实的Qwen API密钥）
    rag_system = QwenRAGSystem(
        qwen_api_key=os.getenv("QWEN_API_KEY", ""),
        embedding_model_name="all-MiniLM-L6-v2"
    )
    if not rag_system.qwen_api_key:
        raise ValueError("未找到 QWEN_API_KEY，请在 .env 配置")
    
    # 示例：构建知识库索引
    chunks = rag_system.load_and_chunk_documents("docs/knowledge_base.txt")
    rag_system.build_vector_index(chunks)
    
    # 示例查询
    result = rag_system.query_rag_system("RAG系统的工作原理是什么？")
    print(json.dumps(result, indent=2, ensure_ascii=False))
