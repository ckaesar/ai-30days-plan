# 基于Qwen API的大模型调用示例
# 注意：此示例仅展示API调用方式，重点在于理解通用概念

import json
import os
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class QwenAPIClient:
    """Qwen API客户端示例 - 演示大模型API通用调用模式"""
    
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化客户端
        :param api_key: Qwen API密钥
        :param base_url: API基础地址（兼容OpenAI格式）
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages, model="qwen-max", temperature=0.7, max_tokens=500):
        """
        对话补全接口调用
        :param messages: 对话消息列表，格式如 [{"role": "user", "content": "你好"}]
        :param model: 模型名称，如 qwen-max, qwen-turbo
        :param temperature: 温度参数，控制随机性
        :param max_tokens: 最大生成Token数
        :return: API响应结果
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False  # 非流式输出
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
    
    def extract_response(self, api_response):
        """
        从API响应中提取生成内容
        :param api_response: API返回的JSON数据
        :return: 生成的文本内容
        """
        if not api_response or "choices" not in api_response:
            return "响应解析失败"
        
        choice = api_response["choices"][0]
        message = choice.get("message", {})
        return message.get("content", "")

# 使用示例
def demo_qwen_api():
    """演示Qwen API调用流程"""
    
    # 1. 配置API密钥（从环境变量读取）
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("错误：未找到 QWEN_API_KEY 环境变量，请在 .env 文件中配置。")
        return
    
    # 2. 初始化客户端
    client = QwenAPIClient(api_key)
    
    # 3. 构建对话消息
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手，擅长用简洁清晰的语言回答问题。"},
        {"role": "user", "content": "请用三句话解释什么是大语言模型。"}
    ]
    
    # 4. 调用API
    print("正在调用Qwen API...")
    response = client.chat_completion(
        messages=messages,
        model="qwen-max",
        temperature=0.3,  # 较低温度确保回答稳定
        max_tokens=200
    )
    
    # 5. 处理响应
    if response:
        content = client.extract_response(response)
        print("生成的回答:")
        print(content)
        
        # 显示Token使用情况（成本控制关键）
        usage = response.get("usage", {})
        print(f"\nToken使用情况:")
        print(f"输入Token: {usage.get('prompt_tokens', 0)}")
        print(f"输出Token: {usage.get('completion_tokens', 0)}")
        print(f"总Token: {usage.get('total_tokens', 0)}")
    
    return response

# 运行演示（实际执行需要有效的API密钥）
if __name__ == "__main__":
    # 注释掉实际调用，避免在没有密钥时出错
    demo_qwen_api()
    # print("请配置有效的Qwen API密钥以运行此示例。")

# 关键要点（模型无关）：
# 1. API调用基本模式：初始化→构建请求→发送→解析
# 2. 参数调优：temperature、max_tokens影响输出质量和成本
# 3. 错误处理：网络异常、认证失败、参数错误等
# 4. 成本意识：关注Token使用量，优化输入输出长度