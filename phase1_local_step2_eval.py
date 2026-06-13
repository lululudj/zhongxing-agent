# -*- coding: utf-8 -*-
"""
Phase 1 Local Step 2: V4-Pro API evaluation of compact_context quality
Group A: Original text + question -> V4-Pro
Group B: compact_context + question -> V4-Pro (with format explanation)
"""
import json, time, requests, sys

DEEPSEEK_API_KEY = "sk-fdfb139836bb4292ad89700f82a2c99a"
DEEPSEEK_BASE = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"

EXTRACTION_FILE = r"D:\phase1_local_extraction.json"
OUTPUT_FILE = r"D:\phase1_local_eval.json"

MACHINE_LANG_SYSTEM = """你是一个精确的信息提取助手。下方提供的上下文是一种结构化知识表示格式：
[实体名] 属性1,属性2,... | 关系1,关系2,... | 概念说明
每行代表一个实体，第一段是属性值，第二段是关系，第三段是概念/补充说明。
请仅根据这些结构化信息回答问题。如果信息不足，请根据已有信息尽可能回答。"""

PLAIN_SYSTEM = "你是一个精确的问答助手。请仅根据提供的上下文回答问题。"


def call_deepseek(system, context, question):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": "上下文:\n" + context + "\n\n问题: " + question}
    ]
    r = requests.post(
        DEEPSEEK_BASE + "/chat/completions",
        headers={"Authorization": "Bearer " + DEEPSEEK_API_KEY},
        json={"model": MODEL, "messages": msgs, "max_tokens": 500, "temperature": 0.1},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return answer, usage


def run_eval():
    # Load extraction results
    with open(EXTRACTION_FILE, "r", encoding="utf-8") as f:
        ext = json.load(f)

    # Load original texts
    sys.path.insert(0, r"C:\Users\Administrator\zhongxing-prototype")
    from test_finance import LONG_TEXT as FINANCE_TEXT, LONG_QUESTIONS as FINANCE_QUESTIONS
    from test_science import LONG_TEXT as SCIENCE_TEXT, LONG_QUESTIONS as SCIENCE_QUESTIONS

    datasets = [
        ("finance", FINANCE_TEXT, FINANCE_QUESTIONS, ext["finance"]["compact_context"]),
        ("science", SCIENCE_TEXT, SCIENCE_QUESTIONS, ext["science"]["compact_context"]),
    ]

    all_results = {}
    total_cost_input = 0
    total_cost_output = 0

    for label, original_text, questions, compact in datasets:
        print("\n" + "=" * 60)
        print("[" + label + "] " + str(len(questions)) + " questions")
        print("Original: " + str(len(original_text)) + " chars")
        print("Compact: " + str(len(compact)) + " chars")
        print("=" * 60)

        results = []
        for i, q in enumerate(questions):
            print("\n  Q" + str(i+1) + ": " + q[:60] + "...")

            # Group A: original text
            print("    A (original)...")
            a_ans, a_usage = call_deepseek(PLAIN_SYSTEM, original_text, q)
            a_input = a_usage.get("prompt_tokens", 0)
            a_output = a_usage.get("completion_tokens", 0)
            print("    -> " + a_ans[:80] + "...")

            # Group B: compact_context (machine language)
            print("    B (compact)...")
            b_ans, b_usage = call_deepseek(MACHINE_LANG_SYSTEM, compact, q)
            b_input = b_usage.get("prompt_tokens", 0)
            b_output = b_usage.get("completion_tokens", 0)
            print("    -> " + b_ans[:80] + "...")

            total_cost_input += a_input + b_input
            total_cost_output += a_output + b_output

            results.append({
                "q": q,
                "A_answer": a_ans,
                "A_input_tokens": a_input,
                "B_answer": b_ans,
                "B_input_tokens": b_input,
            })

            time.sleep(0.5)

        # Calculate token ratio
        total_a_input = sum(r["A_input_tokens"] for r in results)
        total_b_input = sum(r["B_input_tokens"] for r in results)
        token_ratio = total_b_input / max(total_a_input, 1) * 100

        all_results[label] = {
            "results": results,
            "original_len": len(original_text),
            "compact_len": len(compact),
            "char_ratio": round(len(original_text) / max(len(compact), 1), 1),
            "total_a_input_tokens": total_a_input,
            "total_b_input_tokens": total_b_input,
            "token_ratio_pct": round(token_ratio, 1),
        }

        print("\n  [" + label + "] Token ratio: " + str(round(token_ratio, 1)) + "%")
        print("  A total input: " + str(total_a_input) + " tokens")
        print("  B total input: " + str(total_b_input) + " tokens")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("Saved to: " + OUTPUT_FILE)
    for label in all_results:
        r = all_results[label]
        print(label + ": char_ratio=" + str(r["char_ratio"]) + ":1, token_ratio=" + str(r["token_ratio_pct"]) + "%")
    print("Total API tokens: input=" + str(total_cost_input) + " output=" + str(total_cost_output))
    print("=" * 60)


if __name__ == "__main__":
    run_eval()
