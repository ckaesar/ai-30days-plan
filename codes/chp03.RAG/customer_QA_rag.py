#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
客服知识库RAG应用示例
基于企业文档构建智能问答系统
"""
import os
import requests
from typing import Dict, Any
from dotenv import load_dotenv
from rag_example import QwenRAGSystem

load_dotenv()

class CustomerServiceRAG:
    """客服知识库RAG应用"""
    
    def __init__(self, knowledge_base_dir: str):
        """初始化客服知识库"""
        self.knowledge_base_dir = knowledge_base_dir
        self.rag_system = QwenRAGSystem(
            qwen_api_key=os.getenv("QWEN_API_KEY", ""),
            embedding_model_name="BGE-large-zh"  # 中文优化模型
        )
        if not self.rag_system.qwen_api_key:
            raise ValueError("未找到 QWEN_API_KEY，请在 .env 配置")
        
        # 加载知识库文档
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载客服知识库文档"""
        import glob
        
        all_chunks = []
        doc_files = glob.glob(f"{self.knowledge_base_dir}/*.txt") + \
                    glob.glob(f"{self.knowledge_base_dir}/*.md")
        
        for file_path in doc_files:
            chunks = self.rag_system.load_and_chunk_documents(file_path)
            all_chunks.extend(chunks)
        
        # 构建向量索引
        self.rag_system.build_vector_index(all_chunks, "./customer_service_db")
        print(f"客服知识库加载完成：{len(all_chunks)}个文档片段")
    
    def answer_customer_query(self, query: str) -> Dict[str, Any]:
        """回答客户咨询"""
        # 1. 检索相关上下文
        context = self.rag_system.retrieve_relevant_context(query)
        
        # 2. 构建客服专用提示
        customer_prompt = f"""
你是专业的客服助手，基于以下公司政策文档回答客户问题：

【相关政策文档】
{context}

【客户问题】
{query}

回答要求：
1. 严格基于公司政策，不得随意承诺
2. 如果政策不明确，建议转接人工客服
3. 保持礼貌专业，使用客户易懂的语言
4. 必要时提供具体操作步骤
"""
        
        # 3. 调用Qwen生成回答
        payload = {
            "model": os.getenv("LLM_MODEL", "qwen-max"),
            "messages": [{"role": "user", "content": customer_prompt}],
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "400"))
        }
        
        try:
            response = requests.post(
                f"{self.rag_system.base_url}/chat/completions",
                headers=self.rag_system.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            answer = f"抱歉，系统暂时无法生成回答，请稍后再试。（错误：{e}）"
        
        # 4. 添加客服标准话术
        standardized_answer = f"{answer}\n\n---\n如有进一步问题，请随时联系！"
        
        return {
            "query": query,
            "answer": standardized_answer,
            "confidence": self._calculate_confidence(context)
        }
    
    def _calculate_confidence(self, context: str) -> float:
        """计算回答置信度"""
        # 简化示例：基于上下文长度和相关性评分
        if len(context) < 100:
            return 0.3
        elif len(context) < 500:
            return 0.6
        else:
            return 0.8

# 使用示例
if __name__ == "__main__":
    # 初始化客服系统
    customer_service = CustomerServiceRAG("docs")
    
    # 示例客户咨询
    queries = [
        "退货流程是什么？",
        "产品保修期多久？",
        "如何修改订单信息？"
    ]
    
    for query in queries:
        result = customer_service.answer_customer_query(query)
        print(f"问题: {query}")
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['confidence']:.2f}")
        print("-" * 50)
