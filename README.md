# ai-30days-plan
AI技能提升三十天计划

## 环境配置（.env）
项目中的大模型调用使用 OpenAI 兼容接口方式，所有基础参数均从项目根目录下的 `.env` 文件读取，便于无侵入切换不同厂商与模型。

- QWEN_API_KEY：接口访问凭证（务必保密，默认 `.gitignore` 已忽略 `.env`）
- LLM_BASE_URL：OpenAI 兼容的基础地址（如 Qwen 兼容端点）
- LLM_MODEL：模型名称（如 `qwen-max`）
- LLM_TEMPERATURE：采样温度（0–1，越低越稳定）
- LLM_MAX_TOKENS：最大输出 Token 数

示例（请根据实际服务商与密钥进行替换）：

```
QWEN_API_KEY=YOUR_QWEN_API_KEY
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=200
```

运行示例（以 chp00 为例）：

```
python codes/chp00.baseModel/chat.py
```

说明：
- `.env` 已加入 `.gitignore`，请勿提交真实密钥。
- 若切换到其他兼容厂商，仅需调整 `LLM_BASE_URL` 与 `LLM_MODEL` 即可配合新的 API Key 使用。

## 学习计划（plans ↔ codes）
约定：`DayN` 对应 `codes/chp0(N-1)`，后续章节按此规则依次递进。

| Day | 学习主题 | 计划文档（plans） | 对应代码（codes） |
| --- | --- | --- | --- |
| Day1 | 大模型基础 | [plans/Day1-大模型基础.html](plans/Day1-大模型基础.html) | [codes/chp00.baseModel/](codes/chp00.baseModel/) |
| Day2 | Prompt Engineering | [plans/Day2-Prompt-Engineering.html](plans/Day2-Prompt-Engineering.html) | [codes/chp01.PromptEngineering/](codes/chp01.PromptEngineering/) |
| Day3 | Function Calling | [plans/Day3-Function-Calling.html](plans/Day3-Function-Calling.html) | [codes/chp02.FunctionCalling/](codes/chp02.FunctionCalling/) |
