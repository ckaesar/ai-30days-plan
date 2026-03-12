# Agent记忆管理实现
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from reActAgent import QwenReActAgent

load_dotenv()

class AgentMemoryManager:
    """Agent记忆管理系统"""
    
    def __init__(self, max_short_term_memory: int = 10, long_term_storage_path: str = "agent_memory.json"):
        """初始化记忆管理器"""
        self.short_term_memory = []  # 短期记忆：当前对话上下文
        self.long_term_storage_path = long_term_storage_path
        self.max_short_term_memory = max_short_term_memory
        
        # 从文件加载长期记忆（如果存在）
        self.long_term_memory = self.load_long_term_memory()
    
    def load_long_term_memory(self) -> Dict[str, Any]:
        """从文件加载长期记忆"""
        try:
            if os.path.exists(self.long_term_storage_path):
                with open(self.long_term_storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载长期记忆失败: {e}")
        return {
            "user_preferences": {},
            "conversation_history": [],
            "task_patterns": {},
            "error_corrections": []
        }
    
    def save_long_term_memory(self):
        """保存长期记忆到文件"""
        try:
            with open(self.long_term_storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存长期记忆失败: {e}")
    
    def add_to_short_term_memory(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """添加消息到短期记忆"""
        memory_item = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            memory_item["metadata"] = metadata
        
        self.short_term_memory.append(memory_item)
        
        # 控制短期记忆大小
        if len(self.short_term_memory) > self.max_short_term_memory:
            self.short_term_memory = self.short_term_memory[-self.max_short_term_memory:]
    
    def get_context_for_llm(self, max_tokens: int = 3000) -> List[Dict[str, str]]:
        """获取LLM上下文（结合短期和长期记忆）"""
        # 提取短期记忆
        context = []
        for item in self.short_term_memory:
            context.append({"role": item["role"], "content": item["content"]})
        
        # 根据需要添加相关长期记忆
        relevant_memories = self.retrieve_relevant_memories(context)
        for memory in relevant_memories:
            # 添加记忆作为系统提示或上下文
            context.insert(0, {
                "role": "system",
                "content": f"相关背景记忆: {memory}"
            })
        
        return context
    
    def retrieve_relevant_memories(self, current_context: List[Dict[str, str]]) -> List[str]:
        """基于当前上下文检索相关记忆"""
        # 简化实现：在实际项目中应使用向量检索
        relevant_memories = []
        
        # 从长期记忆中提取用户偏好
        user_prefs = self.long_term_memory.get("user_preferences", {})
        if user_prefs:
            relevant_memories.append(f"用户偏好: {json.dumps(user_prefs, ensure_ascii=False)}")
        
        # 基于对话历史提取任务模式
        recent_conversations = self.long_term_memory.get("conversation_history", [])[-5:]
        if recent_conversations:
            # 分析常见任务模式
            patterns = self.extract_task_patterns(recent_conversations)
            if patterns:
                relevant_memories.append(f"近期任务模式: {patterns}")
        
        return relevant_memories
    
    def extract_task_patterns(self, conversations: List[Dict[str, Any]]) -> str:
        """从对话历史中提取任务模式"""
        # 简化实现
        task_types = []
        for conv in conversations:
            if "weather" in conv.get("query", "").lower():
                task_types.append("天气查询")
            elif "calculate" in conv.get("query", "").lower():
                task_types.append("数学计算")
        
        if task_types:
            return "用户经常进行" + "、".join(set(task_types)) + "类任务"
        return ""
    
    def update_user_preferences(self, preference_type: str, value: Any):
        """更新用户偏好"""
        if "user_preferences" not in self.long_term_memory:
            self.long_term_memory["user_preferences"] = {}
        
        self.long_term_memory["user_preferences"][preference_type] = value
        self.save_long_term_memory()

# 带记忆的Agent
class MemoryEnabledAgent(QwenReActAgent):
    """具备记忆能力的Agent"""
    
    def __init__(self, api_key: str, model_name: str = "qwen-max"):
        super().__init__(api_key, model_name or os.getenv("LLM_MODEL", "qwen-max"))
        self.memory_manager = AgentMemoryManager()
    
    def process_query(self, user_query: str) -> str:
        """处理用户查询（带记忆）"""
        # 将查询添加到短期记忆
        self.memory_manager.add_to_short_term_memory("user", user_query)
        
        # 获取上下文（包含记忆）
        context = self.memory_manager.get_context_for_llm()
        
        # 调用LLM处理
        messages = context.copy()
        response = self.call_qwen_api(messages, self.define_agent_tools())
        
        if not response:
            return "处理失败"
        
        # 解析响应
        parsed = self.parse_llm_response(response)
        
        # 处理工具调用结果
        if parsed["tool_calls"]:
            tool_results = []
            for tool_call in parsed["tool_calls"]:
                tool_result = self.execute_tool(tool_call["name"], tool_call["arguments"])
                tool_results.append(tool_result)
            
            # 生成最终回答
            final_response = f"已执行工具调用，结果为: {tool_results}"
        else:
            final_response = parsed["content"]
        
        # 将Agent回答添加到记忆
        self.memory_manager.add_to_short_term_memory("assistant", final_response)
        
        # 更新长期记忆（如有需要）
        self.update_memory_based_on_conversation(user_query, final_response)
        
        return final_response
    
    def update_memory_based_on_conversation(self, user_query: str, agent_response: str):
        """基于对话更新长期记忆"""
        # 记录对话历史
        conversation_record = {
            "query": user_query,
            "response": agent_response,
            "timestamp": datetime.now().isoformat()
        }
        
        if "conversation_history" not in self.memory_manager.long_term_memory:
            self.memory_manager.long_term_memory["conversation_history"] = []
        
        self.memory_manager.long_term_memory["conversation_history"].append(conversation_record)
        
        # 控制历史记录大小
        if len(self.memory_manager.long_term_memory["conversation_history"]) > 100:
            self.memory_manager.long_term_memory["conversation_history"] =                 self.memory_manager.long_term_memory["conversation_history"][-100:]
        
        # 保存记忆
        self.memory_manager.save_long_term_memory()

# 演示带记忆的Agent
def demo_memory_agent():
    """演示带记忆能力的Agent"""
    API_KEY = os.getenv("QWEN_API_KEY")
    if not API_KEY:
        print("错误：未找到 QWEN_API_KEY，请在 .env 配置")
        return
    
    agent = MemoryEnabledAgent(API_KEY, os.getenv("LLM_MODEL", "qwen-max"))
    
    # 多轮对话演示
    conversation = [
        "我喜欢喝美式咖啡",
        "今天北京天气怎么样？",
        "帮我计算一下15%的小费，账单是200元",
        "记住我更喜欢下午喝咖啡"
    ]
    
    for query in conversation:
        print(f"用户: {query}")
        response = agent.process_query(query)
        print(f"Agent: {response}")
        print("-" * 50)

if __name__ == "__main__":
    demo_memory_agent()
