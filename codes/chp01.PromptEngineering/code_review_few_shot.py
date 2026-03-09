def build_prompt():
    header = (
        "系统指令：你是资深代码审查助手。请使用一致的结构化格式给出审查结果，"
        "输出应包含 items（列表）、summary（简要总结）、overall_score（0-10）。\n\n"
        "输出格式示例：\n"
        "{\n"
        "  \"items\": [\n"
        "    {\"type\": \"问题类型\", \"description\": \"问题描述\", \"suggestion\": \"修改建议\", \"severity\": \"high/medium/low\"}\n"
        "  ],\n"
        "  \"summary\": \"整体评价\",\n"
        "  \"overall_score\": 0\n"
        "}\n"
        "请严格保持字段名与层级一致，描述简洁可执行。"
    )

    example1 = (
        "示例1（性能问题）：\n"
        "源码片段：\n"
        "def count_unique(items):\n"
        "    result = 0\n"
        "    for i in range(len(items)):\n"
        "        if items[i] not in items[:i]:\n"
        "            result += 1\n"
        "    return result\n\n"
        "审查输出：\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"type\": \"性能问题\",\n"
        "      \"description\": \"在循环内使用切片与 in 检查导致 O(n^2) 时间复杂度，列表切片也产生额外开销。\",\n"
        "      \"suggestion\": \"改用集合去重或直接使用 set 统计唯一元素，例如：return len(set(items))\",\n"
        "      \"severity\": \"medium\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": \"存在低效去重实现，可通过集合显著优化。\",\n"
        "  \"overall_score\": 7\n"
        "}"
    )

    example2 = (
        "示例2（安全问题）：\n"
        "源码片段：\n"
        "def run(expr):\n"
        "    return eval(expr)\n\n"
        "审查输出：\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"type\": \"安全漏洞\",\n"
        "      \"description\": \"直接对外部输入使用 eval 存在代码注入风险。\",\n"
        "      \"suggestion\": \"避免 eval；若必须解析表达式，使用安全解析库或受限执行环境（如 ast.literal_eval 仅支持字面量）。\",\n"
        "      \"severity\": \"high\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": \"高风险安全隐患，应立即替换为安全解析方案。\",\n"
        "  \"overall_score\": 3\n"
        "}"
    )

    example3 = (
        "示例3（潜在Bug/可维护性）：\n"
        "源码片段：\n"
        "def append_item(x, bucket=[]):\n"
        "    bucket.append(x)\n"
        "    return bucket\n\n"
        "審查输出：\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"type\": \"潜在Bug\",\n"
        "      \"description\": \"使用可变对象作为函数缺省参数会在多次调用间共享同一列表，导致意外数据累计。\",\n"
        "      \"suggestion\": \"将缺省参数设为 None 并在函数内部初始化，例如：def f(x, bucket=None): bucket = bucket or []\",\n"
        "      \"severity\": \"medium\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": \"存在可变缺省参数问题，影响可预测性与维护性。\",\n"
        "  \"overall_score\": 6\n"
        "}"
    )

    guidance = (
        "请在接收到新源码后，严格参考以上示例的结构与风格，"
        "识别主要问题并给出可执行建议，控制总字数在300字以内。"
    )

    return f"{header}\n\n{example1}\n\n{example2}\n\n{example3}\n\n{guidance}"


def main():
    prompt = build_prompt()
    print(prompt)


if __name__ == "__main__":
    main()
