# -*- coding: utf-8 -*-
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
SEP = "=" * 60


def run_extraction_only(text, questions, label=""):
    config = {
        "extractor_model": "qwen2.5:1.5b",
        "schema_model": "qwen2.5:3b",
        "num_extractors": 3,
    }

    print("\n" + SEP)
    print("[" + label + "] text=" + str(len(text)) + " questions=" + str(len(questions)))
    print(SEP)

    t0 = time.time()

    # Layer 0
    print("[1/5] schema...")
    schema = discover_schema(text, config["schema_model"])
    print("  type=" + schema["text_type"] + " entities=" + str(schema["entity_types"][:3]))

    # Pass 1
    print("[2/5] skeleton x" + str(config["num_extractors"]) + "...")
    extractions = run_skeleton_extractors(text, schema, config["extractor_model"], config["num_extractors"])

    # RRF
    print("[3/5] RRF fuse...")
    fusion = rrf_fuse(extractions, config["schema_model"], len(text))
    if isinstance(fusion["fused_context"], list):
        fusion["fused_context"] = {"E": fusion["fused_context"]}

    ent_count = len(fusion["fused_context"].get("E", []))
    print("  " + str(ent_count) + " entities, json=" + str(fusion["json_len"]) + " compact=" + str(fusion["compact_len"]))

    # gap + cooc
    print("  gap check...")
    gap = entity_gap_check(text, fusion["fused_context"], config["schema_model"])
    if gap["missing"]:
        print("  missing: " + str(gap["missing"]) + " -> +" + str(gap["supplement_count"]))
        existing_E = fusion["fused_context"].get("E", [])
        if isinstance(existing_E, list):
            existing_E.extend(gap.get("supplements", []))
    cooc = cooccurrence_scan(text, fusion["fused_context"])
    if cooc["added_rels"] > 0:
        print("  co-occurrence: +" + str(cooc["added_rels"]))

    # relevance
    skeleton_ents = fusion["fused_context"].get("E", [])
    pass2_names, skeleton_only_names, concepts = determine_relevant_entities(
        questions, skeleton_ents, fusion.get("rrf_ranking", []), text)
    print("  pass2=" + str(len(pass2_names)) + " skeleton_only=" + str(len(skeleton_only_names)) + " concepts=" + str(len(concepts)))

    # Pass 2
    print("[4/5] detail extract (" + str(len(pass2_names)) + " ents + " + str(len(concepts)) + " concepts)...")
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
    print("[5/5] done! " + str(round(elapsed, 1)) + "s ratio=" + str(round(compact_ratio, 1)) + ":1")
    print("---preview(500)---")
    print(compact_str[:500])
    print("---end---")

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
        r = requests.get(OLLAMA_BASE + "/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print("Ollama: " + ", ".join(models))
    except Exception as e:
        print("[ERROR] Ollama: " + str(e))
        sys.exit(1)

    finance_result = run_extraction_only(FINANCE_TEXT, FINANCE_QUESTIONS, "finance")
    science_result = run_extraction_only(SCIENCE_TEXT, SCIENCE_QUESTIONS, "science")

    output = {"finance": finance_result, "science": science_result}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nSAVED to " + OUTPUT_FILE)
    print("finance: " + str(finance_result["original_len"]) + "->" + str(finance_result["compact_len"]) + " (" + str(finance_result["compact_ratio"]) + ":1)")
    print("science: " + str(science_result["original_len"]) + "->" + str(science_result["compact_len"]) + " (" + str(science_result["compact_ratio"]) + ":1)")
