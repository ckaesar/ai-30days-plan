# Qwen API Agent基础实现：任务分解与工具调用
import json
import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class QwenReActAgent:
    """基于ReAct范式的Qwen智能体"""
    
    def __init__(self, api_key: str, model_name: str = "qwen-max"):
        """初始化Agent配置"""
        self.api_key = api_key
        self.model_name = model_name or os.getenv("LLM_MODEL", "qwen-max")
        self.base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 定义可用工具集
        self.tools = {
            "weather": self.get_weather_tool,
            "calculator": self.calculate_tool,
            "time": self.get_current_time,
            "search": self.search_web_tool
        }
        
        # 对话历史（短期记忆）
        self.conversation_history = []
        
    def get_weather_tool(self, location: str, unit: str = "celsius") -> str:
        """模拟天气查询工具"""
        # 实际项目中应调用真实天气API
        weather_data = {
            "北京": {"temperature": 15, "condition": "晴朗", "humidity": "45%"},
            "上海": {"temperature": 18, "condition": "多云", "humidity": "65%"},
            "广州": {"temperature": 25, "condition": "小雨", "humidity": "85%"}
        }
        
        if location in weather_data:
            data = weather_data[location]
            return json.dumps({
                "location": location,
                "temperature": data["temperature"],
                "unit": unit,
                "condition": data["condition"],
                "humidity": data["humidity"],
                "timestamp": datetime.now().isoformat()
            })
        return json.dumps({
            "error": "城市不支持",
            "location": location,
            "suggestion": "请检查城市名称"
        })
    
    def calculate_tool(self, expression: str) -> str:
        """计算器工具"""
        try:
            # 安全评估数学表达式
            allowed_chars = "0123456789+-*/(). "
            if any(char not in allowed_chars for char in expression):
                return json.dumps({"error": "表达式包含不安全字符"})
            
            result = eval(expression)
            return json.dumps({
                "expression": expression,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return json.dumps({
                "error": f"计算失败: {e}",
                "expression": expression
            })
    
    def get_current_time(self) -> str:
        """获取当前时间工具"""
        return json.dumps({
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": datetime.now().isoformat()
        })
    
    def search_web_tool(self, query: str, max_results: int = 5) -> str:
        """模拟网页搜索工具"""
        # 实际项目中应调用真实搜索引擎API
        search_results = [
            {"title": "Agent技术发展趋势分析", "url": "https://example.com/agent-trends", "summary": "2024年Agent技术进入快速发展期"},
            {"title": "多智能体协作系统设计", "url": "https://example.com/multi-agent", "summary": "多Agent系统在复杂任务中表现优异"}
        ]
        return json.dumps({
            "query": query,
            "results": search_results[:max_results],
            "timestamp": datetime.now().isoformat()
        })
    
    def call_qwen_api(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用Qwen API（支持工具调用）"""
        endpoint = f"{self.base_url}/chat/completions"
        
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        except ValueError:
            temperature = 0.3
        try:
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1000"))
        except ValueError:
            max_tokens = 1000

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            return None
    
    def define_agent_tools(self) -> List[Dict[str, Any]]:
        """定义Agent可用的工具列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名称"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "数学计算工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "数学表达式"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "获取当前时间",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "网页搜索工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"},
                            "max_results": {"type": "integer", "description": "最大结果数"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析LLM响应，提取工具调用信息"""
        if not response:
            return {"error": "API响应为空"}
        
        message = response["choices"][0]["message"]
        result = {
            "content": message.get("content", ""),
            "tool_calls": []
        }
        
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                func_info = tool_call["function"]
                result["tool_calls"].append({
                    "name": func_info["name"],
                    "arguments": json.loads(func_info["arguments"]),
                    "id": tool_call["id"]
                })
        
        return result
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用"""
        if tool_name not in self.tools:
            return json.dumps({"error": f"工具不存在: {tool_name}"})
        
        try:
            tool_func = self.tools[tool_name]
            
            # 根据工具函数签名提取参数
            if tool_name == "weather":
                return tool_func(arguments.get("location"), arguments.get("unit", "celsius"))
            elif tool_name == "calculator":
                return tool_func(arguments.get("expression"))
            elif tool_name == "time":
                return tool_func()
            elif tool_name == "search":
                return tool_func(arguments.get("query"), arguments.get("max_results", 5))
            else:
                return json.dumps({"error": "参数解析失败"})
        except Exception as e:
            return json.dumps({"error": f"工具执行异常: {e}"})
    
    def react_cycle(self, user_query: str, max_steps: int = 5) -> str:
        """执行ReAct循环解决用户问题"""
        print(f"开始处理用户查询: {user_query}")
        print("=" * 50)
        
        # 初始化工具列表
        tools = self.define_agent_tools()
        
        # 构建初始消息
        messages = [
            {"role": "user", "content": user_query}
        ]
        
        for step in range(max_steps):
            print(f"第{step+1}步:")
            
            # 调用Qwen API（包含工具描述）
            response = self.call_qwen_api(messages, tools)
            if not response:
                return "API调用失败，请检查网络和API密钥"
            
            # 解析响应
            parsed = self.parse_llm_response(response)
            messages.append(response["choices"][0]["message"])
            
            # 检查是否有工具调用
            if parsed["tool_calls"]:
                for tool_call in parsed["tool_calls"]:
                    print(f"  调用工具: {tool_call['name']}")
                    print(f"  参数: {tool_call['arguments']}")
                    
                    # 执行工具
                    tool_result = self.execute_tool(tool_call["name"], tool_call["arguments"])
                    print(f"  工具结果: {tool_result}")
                    
                    # 将工具结果添加到对话历史
                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call["id"]
                    })
            else:
                # 没有工具调用，直接返回答案
                print(f"  直接回答: {parsed['content']}")
                return parsed["content"]
        
        return "达到最大步数限制，任务未完成"

# Agent使用示例
def demo_basic_agent():
    """演示基础Agent功能"""
    API_KEY = os.getenv("QWEN_API_KEY")
    if not API_KEY:
        print("错误：未找到 QWEN_API_KEY，请在 .env 配置")
        return
    
    agent = QwenReActAgent(API_KEY, os.getenv("LLM_MODEL", "qwen-max"))
    
    # 测试简单查询
    test_queries = [
        "北京今天的天气怎么样？",
        "计算(15 + 27) * 3.5的结果",
        "现在是什么时间？",
        "帮我搜索一下Agent的最新发展",
        "先查上海明天的天气，然后计算温度乘以2"
    ]
    
    for query in test_queries:
        print(f"查询: {query}")
        result = agent.react_cycle(query)
        print(f"最终回答: {result}")
        print("-" * 50)

if __name__ == "__main__":
    demo_basic_agent()
