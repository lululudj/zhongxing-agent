# -*- coding: utf-8 -*-
"""深度诊断qwen3.5:4b - 看完整API返回"""
import requests, json

OLLAMA_BASE = "http://localhost:11434"

test_text = "美联储主席鲍威尔表示通胀降温，美联储决定降息25个基点。"

prompt = '提取文本中实体名字和关系→JSON。输出: {"E":[{"n":"名","R":[...]}]}\n文本: ' + test_text

# 测试1: 标准调用，打印完整返回
print("=== 测试1: qwen3.5:4b 标准调用 ===")
try:
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json={
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }, timeout=120)
    resp = r.json()
    print(f"完整响应keys: {list(resp.keys())}")
    msg = resp.get("message", {})
    print(f"message keys: {list(msg.keys())}")
    print(f"content类型: {type(msg.get('content'))}, 长度: {len(msg.get('content',''))}")
    print(f"content: [{msg.get('content','')[:500]}]")
    print(f"role: {msg.get('role')}")
    if msg.get('tool_calls'):
        print(f"tool_calls: {msg['tool_calls']}")
    print(f"done_reason: {resp.get('done_reason')}")
    print(f"eval_count: {resp.get('eval_count')}")
    print(f"total_duration: {resp.get('total_duration')}")
except Exception as e:
    print(f"错误: {e}")

print()

# 测试2: 加 /no_think
print("=== 测试2: qwen3.5:4b 加/no_think ===")
try:
    r2 = requests.post(f"{OLLAMA_BASE}/api/chat", json={
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
        "stream": False
    }, timeout=120)
    resp2 = r2.json()
    content2 = resp2.get("message", {}).get("content", "")
    print(f"content长度: {len(content2)}")
    print(f"content: [{content2[:500]}]")
except Exception as e:
    print(f"错误: {e}")

print()

# 测试3: 简单问答看4b能不能说话
print("=== 测试3: qwen3.5:4b 简单问答 ===")
try:
    r3 = requests.post(f"{OLLAMA_BASE}/api/chat", json={
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "1+1等于几？只回答数字"}],
        "stream": False
    }, timeout=60)
    content3 = r3.json().get("message", {}).get("content", "")
    print(f"content: [{content3[:500]}]")
except Exception as e:
    print(f"错误: {e}")

print()

# 测试4: qwen3.5:4b 带 think 标签检查
print("=== 测试4: qwen3.5:4b 检查think标签 ===")
try:
    r4 = requests.post(f"{OLLAMA_BASE}/api/chat", json={
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "说一个关于猫的成语"}],
        "stream": False
    }, timeout=60)
    resp4 = r4.json()
    content4 = resp4.get("message", {}).get("content", "")
    has_think = "<think>" in content4 or "</think>" in content4
    print(f"有think标签: {has_think}")
    print(f"content长度: {len(content4)}")
    print(f"前500字: [{content4[:500]}]")
    print(f"后200字: [{content4[-200:]}]")
except Exception as e:
    print(f"错误: {e}")
