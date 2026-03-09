import os
import json
import math
import ast
import operator
import requests
from dotenv import load_dotenv
load_dotenv()


class MultiFunctionDemo:
    def __init__(self, api_key: str, base_url: str | None = None, model: str = "qwen-max"):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("LLM_MODEL", "qwen-max")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def define_multiple_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "搜索最新网页信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"},
                            "max_results": {"type": "integer", "description": "最大结果数"}
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "数学计算工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "数学表达式"}
                        },
                        "required": ["expression"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_units",
                    "description": "单位转换工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number", "description": "数值"},
                            "from_unit": {"type": "string", "description": "原单位"},
                            "to_unit": {"type": "string", "description": "目标单位"}
                        },
                        "required": ["value", "from_unit", "to_unit"],
                        "additionalProperties": False
                    }
                }
            }
        ]

    def call_with_tools(self, user_query: str, tools: list):
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_query}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 400
        }
        try:
            r = requests.post(endpoint, headers=self.headers, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            return None

    def execute_search_web(self, query: str, max_results: int = 3) -> str:
        results = [{"title": f"结果{i+1}", "snippet": f"{query} 的相关简述{i+1}"} for i in range(max_results)]
        return json.dumps({"query": query, "results": results})

    def execute_calculate(self, expression: str) -> str:
        allowed_nodes = {
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv, ast.Call, ast.Name
        }
        allowed_funcs = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log}
        try:
            node = ast.parse(expression, mode="eval")
            if not all(type(n) in allowed_nodes for n in ast.walk(node)):
                return json.dumps({"error": "不支持的表达式"})
            def eval_node(n):
                if isinstance(n, ast.Expression):
                    return eval_node(n.body)
                if isinstance(n, ast.Num):
                    return n.n
                if isinstance(n, ast.BinOp):
                    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
                           ast.FloorDiv: operator.floordiv}
                    return ops[type(n.op)](eval_node(n.left), eval_node(n.right))
                if isinstance(n, ast.UnaryOp):
                    if isinstance(n.op, ast.USub):
                        return -eval_node(n.operand)
                    if isinstance(n.op, ast.UAdd):
                        return +eval_node(n.operand)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in allowed_funcs:
                    args = [eval_node(a) for a in n.args]
                    return allowed_funcs[n.func.id](*args)
                if isinstance(n, ast.Name) and n.id in {"pi", "e"}:
                    return {"pi": math.pi, "e": math.e}[n.id]
                raise ValueError("不支持的节点")
            value = eval_node(node)
            return json.dumps({"expression": expression, "value": value})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def execute_convert_units(self, value: float, from_unit: str, to_unit: str) -> str:
        u = (from_unit or "").lower()
        v = (to_unit or "").lower()
        if u in {"f", "fahrenheit"} and v in {"c", "celsius"}:
            res = (value - 32) * 5.0 / 9.0
            return json.dumps({"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": res})
        if u in {"c", "celsius"} and v in {"f", "fahrenheit"}:
            res = value * 9.0 / 5.0 + 32
            return json.dumps({"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": res})
        return json.dumps({"error": "暂不支持的单位"})

    def demo(self):
        tools = self.define_multiple_tools()
        queries = [
            "计算圆的面积，半径是5厘米",
            "今天比特币价格多少？",
            "把100华氏度转换成摄氏度",
            "请帮我写一段关于AI伦理的摘要"
        ]
        print("=== 多函数选择演示 ===")
        for q in queries:
            print(f"用户查询: {q}")
            first = self.call_with_tools(q, tools)
            if not first:
                print("调用失败\n")
                continue
            msg = first["choices"][0]["message"]
            if "tool_calls" not in msg:
                print(f"直接回答: {msg.get('content','')}\n")
                continue
            call = msg["tool_calls"][0]
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            if name == "search_web":
                result = self.execute_search_web(args.get("query", ""), int(args.get("max_results", 3) or 3))
            elif name == "calculate":
                result = self.execute_calculate(args.get("expression", ""))
            elif name == "convert_units":
                result = self.execute_convert_units(float(args.get("value", 0)), args.get("from_unit", ""), args.get("to_unit", ""))
            else:
                result = json.dumps({"error": "未知工具"})
            second_payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "content": result, "tool_call_id": call["id"]}
                ],
                "temperature": 0.7,
                "max_tokens": 400
            }
            second = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=second_payload)
            if second.status_code == 200:
                print(f"最终回答: {second.json()['choices'][0]['message']['content']}\n")
            else:
                print(f"工具结果: {result}\n")


def main():
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("错误：未找到 QWEN_API_KEY，请在 .env 配置")
        return
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL", "qwen-max")
    demo = MultiFunctionDemo(api_key, base_url, model)
    demo.demo()


if __name__ == "__main__":
    main()
