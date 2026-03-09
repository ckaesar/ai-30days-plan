import json
import os
import requests
from dotenv import load_dotenv
load_dotenv()

class PromptEngineeringDemo:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.base_url = base_url
        self.default_model = os.getenv("LLM_MODEL", "qwen-max")
        try:
            self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        except ValueError:
            self.default_temperature = 0.7
        try:
            self.default_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "500"))
        except ValueError:
            self.default_max_tokens = 500
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def call_qwen_api(self, messages, model=None, temperature=None, max_tokens=None):
        endpoint = f"{self.base_url}/chat/completions"
        if model is None:
            model = self.default_model
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
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
    
    def demo_role_playing(self):
        print("=== 技巧1：角色设定示例 ===")
        
        messages = [
            {
                "role": "system",
                "content": "你是一位资深的技术面试官，专注于后端开发和系统设计。你的风格是专业、严谨但友好。"
            },
            {
                "role": "user",
                "content": "请为一个有3年经验的Java工程师设计一道系统设计面试题，并给出评估要点。"
            }
        ]
        
        response = self.call_qwen_api(messages, temperature=0.3, max_tokens=300)
        if response:
            content = self.extract_content(response)
            print(content)
        return response
    
    def demo_chain_of_thought(self):
        print("\n=== 技巧2：思维链示例 ===")
        
        messages = [
            {
                "role": "user",
                "content": """请逐步思考以下问题：
一家电商网站有1000万用户，日订单量10万。现在需要设计一个优惠券系统，支持：
1. 多种优惠类型（满减、折扣、免运费）
2. 精准投放（用户分层、商品分类）
3. 防刷和限流

请分步骤设计系统架构，并解释关键决策原因。"""
            }
        ]
        
        response = self.call_qwen_api(messages, temperature=0.2, max_tokens=800)
        if response:
            content = self.extract_content(response)
            print(content)
        return response
    
    def demo_structured_output(self):
        print("\n=== 技巧3：结构化输出示例 ===")
        
        messages = [
            {
                "role": "system",
                "content": "你是一个代码审查助手，擅长发现代码中的问题并提供改进建议。"
            },
            {
                "role": "user",
                "content": """请分析以下Python函数的潜在问题，并以JSON格式返回：
{
  "function_name": "process_data",
  "issues": [
    {
      "type": "性能问题/安全漏洞/代码风格/潜在bug",
      "description": "详细描述",
      "severity": "high/medium/low",
      "suggestion": "改进建议"
    }
  ],
  "overall_score": 0-10
}

需要分析的函数：
def process_data(data_list):
    result = []
    for item in data_list:
        # 复杂处理逻辑
        processed = item * 2 + 5
        result.append(processed)
    return result"""
            }
        ]
        
        response = self.call_qwen_api(messages, temperature=0.1, max_tokens=600)
        if response:
            content = self.extract_content(response)
            print("结构化输出示例：")
            print(content)
            
            # 尝试解析JSON验证格式正确性
            try:
                parsed = json.loads(content)
                print(f"\n成功解析JSON，共发现 {len(parsed.get('issues', []))} 个问题")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
        return response
    
    def demo_few_shot_learning(self):
        print("\n=== 技巧4：Few-shot学习示例 ===")
        
        messages = [
            {
                "role": "user",
                "content": """请根据示例完成情感分析：

示例1:
输入: "这个产品太棒了，完全超出预期！"
输出: {"sentiment": "positive", "confidence": 0.95, "reason": "包含积极词汇'太棒了''超出预期'"}

示例2:
输入: "服务很差，再也不会购买了。"
输出: {"sentiment": "negative", "confidence": 0.88, "reason": "包含负面词汇'很差''不会再购买'"}

现在请分析:
输入: "质量一般，但价格还算合理。"
输出:"""
            }
        ]
        
        response = self.call_qwen_api(messages, temperature=0.2, max_tokens=300)
        if response:
            content = self.extract_content(response)
            print(content)
        return response
    
    def demo_parameter_tuning(self):
        print("\n=== 技巧5：参数调优对比 ===")
        
        base_prompt = "请用Python实现一个快速排序算法"
        
        print("1. temperature=0.1（稳定输出）:")
        messages = [{"role": "user", "content": base_prompt}]
        response_low = self.call_qwen_api(messages, temperature=0.1, max_tokens=400)
        if response_low:
            content = self.extract_content(response_low)
            print(f"输出长度: {len(content)}字符")
        
        print("\n2. temperature=0.9（创意输出）:")
        response_high = self.call_qwen_api(messages, temperature=0.9, max_tokens=400)
        if response_high:
            content = self.extract_content(response_high)
            print(f"输出长度: {len(content)}字符")
        
        return response_low, response_high
    
    def extract_content(self, api_response):
        if not api_response or "choices" not in api_response:
            return "响应解析失败"
        
        choice = api_response["choices"][0]
        message = choice.get("message", {})
        return message.get("content", "")
    
    def run_all_demos(self):
        print("Prompt Engineering六大核心技巧演示")
        print("=" * 50)
        
        print("注意：以下演示需要有效的Qwen API密钥")
        self.demo_role_playing()
        print("\n演示完成。")

def main():
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("错误：未找到 QWEN_API_KEY，请在 .env 配置")
        return
    base_url = os.getenv("LLM_BASE_URL")
    
    demo = PromptEngineeringDemo(api_key, base_url)
    
    demo.run_all_demos()
    
    print("提示工程示例代码加载完成。")
    print("核心要点：")
    print("1. 角色设定 → 专业领域聚焦")
    print("2. 思维链 → 复杂问题分解")
    print("3. 结构化输出 → 程序化处理")
    print("4. Few-shot学习 → 快速任务适配")
    print("5. 参数调优 → 输出质量控制")
    print("6. 上下文管理 → 对话连贯性保障")

if __name__ == "__main__":
    main()
