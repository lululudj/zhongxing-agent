# -*- coding: utf-8 -*-
"""
Phase 1 本地验证 Step 1: 用zhongxing本地管道提取compact_context
跳过big_brain推理（后续用DeepSeek V4-Pro API评估）
"""
import sys, json, time, requests

sys.path.insert(0, r"C:\Users\Administrator\zhongxing-prototype")

from zhongxing_agent import (
    discover_schema, run_skeleton_extractors, rrf_fuse,
    entity_gap_check, cooccurrence_scan, determine_relevant_entities,
    run_detail_extraction, merge_details, serialize_compact,
    call_ollama, OLLAMA_BASE
)
from test_finance import LONG_TEXT as FINANCE_TEXT, LONG_QUESTIONS as FINANCE_QUESTIONS
from test_science import LONG_TEXT as SCIENCE_TEXT, LONG_QUESTIONS as SCIENCE_QUESTIONS

OUTPUT_FILE = r"D:\phase1_local_extraction.json"


def run_extraction_only(text, questions, label=""):
    """只运行提取管道，不跑big_brain推理"""
    config = {
        "extractor_model": "qwen2.5:1.5b",
        "schema_model": "qwen2.5:3b",
        "num_extractors": 3,
    }

    print(f"\n{'='*60}")
    print(f"[{label}] text={len(text)} questions={len(questions)}")
    print(f"{'='*60}")

    t0 = time.time()

    # Layer 0
    print("[1/5] schema...")
    schema = discover_schema(text, config["schema_model"])
    print(f"  type={schema['text_type']} entities={schema['entity_types'][:3]}")

    # Pass 1
    print(f"[2/5] skeleton x{config['num_extractors']}...")
    extractions = run_skeleton_extractors(text, schema, config["extractor_model"], config["num_extractors"])

    # RRF
    print("[3/5] RRF fuse...")
    fusion = rrf_fuse(extractions, config["schema_model"], len(text))
    if isinstance(fusion["fused_context"], list):
        fusion["fused_context"] = {"E": fusion["fused_context"]}

    ent_count = len(fusion["fused_context"].get("E", []))
    print(f"  {ent_count} entities, json={fusion['json_len']} compact={fusion['compact_len']}")

    # gap + cooc
    print("  gap check...")
    gap = entity_gap_check(text, fusion["fused_context"], config["schema_model"])
    if gap["missing"]:
        print(f"  missing: {gap['missing']} -> +{gap['supplement_count']}")
        existing_E = fusion["fused_context"].get("E", [])
        if isinstance(existing_E, list):
            existing_E.extend(gap.get("supplements", []))
    cooc = cooccurrence_scan(text, fusion["fused_context"])
    if cooc["added_rels"] > 0:
        print(f"  co-occurrence: +{cooc['added_rels']}")

    # relevance
    skeleton_ents = fusion["fused_context"].get("E", [])
    pass2_names, skeleton_only_names, concepts = determine_relevant_entities(
        questions, skeleton_ents, fusion.get("rrf_ranking", []), text)
    print(f"  pass2={len(pass2_names)} skeleton_only={len(skeleton_only_names)} concepts={len(concepts)}")

    # Pass 2
    print(f"[4/5] detail extract ({len(pass2_names)} ents + {len(concepts)} concepts)...")
    detail_results = run_detail_extraction(pass2_names, concepts, skeleton_ents, text, config["extractor_model"])

    # merge
    merged_ents = merge_details(skeleton_ents, detail_results, detail_results.get("concept_details", {}), text)
    fusion["fused_context"] = {"E": merged_ents}
    fusion["compact_context"] = serialize_compact(fusion["fused_context"])

    compact_str = fusion["compact_context"]
    json_str = json.dumps(fusion["fused_context"], ensure_ascii=False)
    compact_ratio = len(text) / max(len(compact_str), 1)
    json_ratio = len(text) / max(len(json_str), 1)

    elapsed = time.time() - t0
    print(f"[5/5] done! {elapsed:.1f}s ratio={compact_ratio:.1f}:1")
    print(f"---preview(500)---\n{compact_str[:500]}\n---end---")

    return {
        "compact_context": compact_str,
        "fused_context": fusion["fused_context"],
        "compact_len": len(compact_str),
        "json_len": len(json_str),
        "original_len": len(text),
        "compact_ratio": round(compact_ratio, 1),
        "json_ratio": round(json_ratio, 1),
        "elapsed": round(elapsed, 1),
        "entity_count": len(merged_ents),
    }


if __name__ == "__main__":
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama: {models}")
    except Exception as e:
        print(f"[ERROR] Ollama: {e}")
        sys.exit(1)

    finance_result = run_extraction_only(FINANCE_TEXT, FINANCE_QUESTIONS, "finance")
    science_result = run_extraction_only(SCIENCE_TEXT, SCIENCE_QUESTIONS, "science")

    output = {"finance": finance_result, "science": science_result}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSAVED to {OUTPUT_FILE}")
    print(f"finance: {finance_result['original_len']}->{finance_result['compact_len']} ({finance_result['compact_ratio']}:1)")
    print(f"science: {science_result['original_len']}->{science_result['compact_len']} ({science_result['compact_ratio']}:1)")
