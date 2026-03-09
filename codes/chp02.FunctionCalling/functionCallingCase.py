# Function Calling基础示例：天气查询工具
import json
import os
from typing import Dict, Any
from datetime import datetime
import requests
from dotenv import load_dotenv
load_dotenv()

class QwenFunctionCallingDemo:
    
    def __init__(self, api_key: str, base_url: str | None = None, model: str = "qwen-max"):
        self.api_key = api_key
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("LLM_MODEL", "qwen-max")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def define_weather_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海、广州"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位，摄氏度或华氏度"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False
                }
            }
        }
    
    def call_qwen_with_tools(self, user_query: str, tools: list) -> Dict[str, Any] | None:
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_query}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 500
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
    
    def execute_weather_function(self, location: str, unit: str = "celsius") -> str:
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
        else:
            return json.dumps({
                "location": location,
                "temperature": "未知",
                "unit": unit,
                "condition": "数据不可用",
                "error": "城市不支持"
            })
    
    def demo_basic_function_calling(self) -> None:
        print("=== 基础Function Calling演示 ===")
        
        weather_tool = self.define_weather_tool()
        tools = [weather_tool]
        
        user_query = "北京今天的天气怎么样？"
        print(f"用户查询: {user_query}")
        
        response = self.call_qwen_with_tools(user_query, tools)
        if not response:
            print("API调用失败")
            return
        
        message = response["choices"][0]["message"]
        if "tool_calls" in message:
            tool_call = message["tool_calls"][0]
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            print(f"模型决定调用工具: {function_name}")
            print(f"生成参数: {arguments}")
            
            if function_name == "get_current_weather":
                location = arguments.get("location")
                unit = arguments.get("unit", "celsius")
                weather_result = self.execute_weather_function(location, unit)
                print(f"工具执行结果: {weather_result}")
                
                second_payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                        {"role": "tool", "content": weather_result, "tool_call_id": tool_call["id"]}
                    ],
                    "temperature": 0.7
                }
                
                final_response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=second_payload
                )
                
                if final_response.status_code == 200:
                    final_content = final_response.json()["choices"][0]["message"]["content"]
                    print(f"最终回答: {final_content}")
        else:
            print("模型决定不调用工具，直接回答")


def main():
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("错误：未找到 QWEN_API_KEY，请在 .env 配置")
        return
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL", "qwen-max")
    demo = QwenFunctionCallingDemo(api_key, base_url, model)
    demo.demo_basic_function_calling()


if __name__ == "__main__":
    main()
