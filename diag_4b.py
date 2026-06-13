# -*- coding: utf-8 -*-
"""诊断qwen3.5:4b原始输出 - 看它到底吐了什么"""
import requests, json

OLLAMA_BASE = "http://localhost:11434"

test_text = """美联储主席鲍威尔在12月议息会议后表示，通胀已明显降温，经济下行风险增加。美联储决定降息25个基点，联邦基金利率降至4.25%-4.5%。这是2024年连续第三次降息，累计降息100个基点。"""

prompt = (
    '提取文本中所有实体的名字和关系→JSON。\n'
    '关系类型: 人物/组织/事件/概念\n'
    '规则:\n'
    '1. 每个实体一条，n写真名(2-4字)\n'
    '2. R写关系词+关键事件\n'
    '3. 只输出JSON\n'
    '输出: {"E":[{"n":"名","R":[...]}]}\n'
    f'文本: {test_text}'
)
system = "提取器。提名+关系。只输出JSON。"

print("=== 测试1: 标准chat (qwen3.5:4b) ===")
r = requests.post(f"{OLLAMA_BASE}/api/chat", json={
    "model": "qwen3.5:4b",
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ],
    "stream": False
}, timeout=120)
resp = r.json()
raw = resp["message"]["content"]
print(f"原始输出({len(raw)}字):")
print(raw[:2000])
print()

# 截断think
if "</think>" in raw:
    after_think = raw.split("</think>")[-1].strip()
    print(f"=== 截断think后({len(after_think)}字) ===")
    print(after_think[:1000])
    print()

# 尝试解析JSON
import re
json_match = re.search(r'\{[\s\S]*\}', after_think if "</think>" in raw else raw)
if json_match:
    try:
        parsed = json.loads(json_match.group())
        print(f"=== JSON解析成功: {len(parsed.get('E', []))} 实体 ===")
        for e in parsed.get('E', [])[:5]:
            print(f"  {e}")
    except:
        print("JSON解析失败")
else:
    print("未找到JSON结构")

print()
print("=== 测试2: 对比 qwen2.5:3b ===")
r2 = requests.post(f"{OLLAMA_BASE}/api/chat", json={
    "model": "qwen2.5:3b",
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ],
    "stream": False
}, timeout=60)
raw2 = r2.json()["message"]["content"]
print(f"原始输出({len(raw2)}字):")
print(raw2[:1000])
