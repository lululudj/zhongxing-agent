# -*- coding: utf-8 -*-
"""
Phase 1 Extract V2: 双模型对比提取 + 三指标评估
- 支持两次运行：3b提取器 vs 4b提取器
- 新增指标：提取耗时 / VRAM峰值 / Schema崩溃率
- 实体质量统计：有属性实体占比 / 关系实体占比
"""
import sys, json, time, requests, threading, subprocess, re, os
sys.path.insert(0, r'C:\Users\Administrator\zhongxing-prototype')
from zhongxing_agent import (
    discover_schema, run_skeleton_extractors, rrf_fuse,
    entity_gap_check, cooccurrence_scan, determine_relevant_entities,
    run_detail_extraction, merge_details, serialize_compact,
    OLLAMA_BASE, call_ollama, try_parse_json
)
import importlib.util

def load_test(path):
    spec = importlib.util.spec_from_file_location('td', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LONG_TEXT, mod.LONG_QUESTIONS

# ============ 配置 ============
EXTRACTOR_MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen2.5:3b'
SCHEMA_MODEL = EXTRACTOR_MODEL
NUM_EXTRACTORS = 2
OUTPUT_DIR = 'D:/'

SEP = '=' * 60

# ============ VRAM监控线程 ============
class VRAMMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mb = 0
        self.running = False
        self._samples = []

    def run(self):
        self.running = True
        while self.running:
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    timeout=3, stderr=subprocess.DEVNULL
                ).decode()
                mb = int(out.strip().split('\n')[0].strip())
                self._samples.append(mb)
                if mb > self.peak_mb:
                    self.peak_mb = mb
            except:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.join(timeout=2)

    def avg_mb(self):
        return sum(self._samples) / max(len(self._samples), 1)

# ============ Schema崩溃率统计 ============
class SchemaStats:
    def __init__(self):
        self.total_calls = 0
        self.failures = 0
        self.failure_details = []

    def record(self, raw_output, label=""):
        self.total_calls += 1
        parsed, ok = try_parse_json(raw_output)
        if not ok:
            self.failures += 1
            self.failure_details.append({
                "label": label,
                "reason": "json_parse_failed",
                "raw_preview": raw_output[:200]
            })
            return parsed, False
        # 检查实体是否缺少n字段
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and not item.get('n'):
                    self.failures += 1
                    self.failure_details.append({
                        "label": label,
                        "reason": "missing_entity_name",
                        "raw_preview": str(item)[:200]
                    })
                    break
        return parsed, True

    def failure_rate(self):
        return self.failures / max(self.total_calls, 1)

# ============ 单次提取运行 ============
def run_extract(text, questions, label, extractor_model, schema_model):
    config = {
        'extractor_model': extractor_model,
        'schema_model': schema_model,
        'num_extractors': NUM_EXTRACTORS
    }

    print('\n' + SEP)
    print('[' + label + '] extractor=' + extractor_model + ' schema=' + schema_model)
    print('  text=' + str(len(text)) + ' chars, questions=' + str(len(questions)))

    # 启动VRAM监控
    vram = VRAMMonitor(interval=0.5)
    vram.start()
    baseline_mb = vram.peak_mb

    schema_stats = SchemaStats()
    phase_times = {}

    t0 = time.time()

    # Phase 1: Schema发现
    pt = time.time()
    print('[1/5] schema...')
    schema = discover_schema(text, config['schema_model'])
    phase_times['schema'] = round(time.time() - pt, 2)
    print('  type=' + schema['text_type'] + ' (' + str(phase_times['schema']) + 's)')

    # Phase 2: 骨架提取
    pt = time.time()
    print('[2/5] skeleton x' + str(config['num_extractors']) + '...')
    exts = run_skeleton_extractors(text, schema, config['extractor_model'], config['num_extractors'])
    phase_times['skeleton'] = round(time.time() - pt, 2)
    print('  (' + str(phase_times['skeleton']) + 's)')

    # Phase 3: RRF融合
    pt = time.time()
    print('[3/5] RRF fuse + gap check + cooc...')
    fusion = rrf_fuse(exts, config['schema_model'], len(text))
    if isinstance(fusion['fused_context'], list):
        fusion['fused_context'] = {'E': fusion['fused_context']}

    # Gap check
    gap = entity_gap_check(text, fusion['fused_context'], config['schema_model'])
    if gap['missing']:
        print('  missing entities: ' + str(gap['missing']))
        E = fusion['fused_context'].get('E', [])
        if isinstance(E, list):
            E.extend(gap.get('supplements', []))

    # Co-occurrence
    cooc = cooccurrence_scan(text, fusion['fused_context'])
    if cooc['added_rels'] > 0:
        print('  co-occ: +' + str(cooc['added_rels']) + ' relations')

    # Relevance filtering
    sk_ents = fusion['fused_context'].get('E', [])
    p2, sk_only, con = determine_relevant_entities(questions, sk_ents, fusion.get('rrf_ranking', []), text)
    print('  p2=' + str(len(p2)) + ' sk=' + str(len(sk_only)) + ' con=' + str(len(con)))
    phase_times['fuse'] = round(time.time() - pt, 2)

    # Phase 4: Detail提取
    pt = time.time()
    print('[4/5] detail extract...')
    det = run_detail_extraction(p2, con, sk_ents, text, config['extractor_model'])
    merged = merge_details(sk_ents, det, det.get('concept_details', {}), text)
    fusion['fused_context'] = {'E': merged}
    phase_times['detail'] = round(time.time() - pt, 2)

    # Phase 5: 序列化
    pt = time.time()
    fusion['compact_context'] = serialize_compact(fusion['fused_context'])
    phase_times['serialize'] = round(time.time() - pt, 2)

    elapsed = time.time() - t0

    # 停止VRAM监控
    vram.stop()

    # 实体质量统计
    cs = fusion['compact_context']
    ents = merged
    ents_with_attrs = 0
    ents_with_rels = 0
    ents_with_concept = 0
    empty_attrs = 0
    for e in ents:
        a = e.get('A', [])
        r = e.get('R', [])
        c = e.get('C', [])
        has_a = bool(a and any(str(x).strip() for x in a))
        has_r = bool(r and any(str(x).strip() for x in r))
        has_c = bool(c and any(str(x).strip() for x in c))
        if has_a:
            ents_with_attrs += 1
        else:
            empty_attrs += 1
        if has_r:
            ents_with_rels += 1
        if has_c:
            ents_with_concept += 1

    cr = len(text) / max(len(cs), 1)
    print('[5/5] done! ' + str(round(elapsed, 1)) + 's')
    print('  compact: ' + str(len(text)) + '->' + str(len(cs)) + ' (' + str(round(cr, 1)) + ':1)')
    print('  entities: ' + str(len(ents)) + ' (with_A=' + str(ents_with_attrs) + ' with_R=' + str(ents_with_rels) + ' with_C=' + str(ents_with_concept) + ' empty_A=' + str(empty_attrs) + ')')
    print('  VRAM peak: ' + str(vram.peak_mb) + ' MB, avg: ' + str(round(vram.avg_mb())) + ' MB')
    print('  Phase times: ' + json.dumps(phase_times))
    print('---preview---')
    print(cs[:600])
    print('---end---')

    return {
        'compact_context': cs,
        'compact_len': len(cs),
        'original_len': len(text),
        'compact_ratio': round(cr, 1),
        'elapsed': round(elapsed, 1),
        'entity_count': len(ents),
        'ents_with_attrs': ents_with_attrs,
        'ents_with_rels': ents_with_rels,
        'ents_with_concept': ents_with_concept,
        'empty_attrs': empty_attrs,
        'vram_peak_mb': vram.peak_mb,
        'vram_avg_mb': round(vram.avg_mb()),
        'phase_times': phase_times,
        'extractor_model': extractor_model,
        'schema_model': schema_model,
    }


if __name__ == '__main__':
    # 检查Ollama模型可用性
    try:
        r = requests.get(OLLAMA_BASE + '/api/tags', timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        print('Ollama models: ' + ', '.join(models))
    except Exception as e:
        print('Ollama connection failed: ' + str(e))
        sys.exit(1)

    # 确认提取器模型已安装
    model_installed = any(EXTRACTOR_MODEL in m for m in models)
    if not model_installed:
        print('Model ' + EXTRACTOR_MODEL + ' not found! Available: ' + str(models))
        print('Run: ollama pull ' + EXTRACTOR_MODEL)
        sys.exit(1)

    # 加载测试数据
    ft, fq = load_test(r'C:\Users\Administrator\zhongxing-prototype\test_finance.py')
    st, sq = load_test(r'C:\Users\Administrator\zhongxing-prototype\test_science.py')

    # 运行提取
    fr = run_extract(ft, fq, 'finance', EXTRACTOR_MODEL, SCHEMA_MODEL)
    sr = run_extract(st, sq, 'science', EXTRACTOR_MODEL, SCHEMA_MODEL)

    # 保存结果
    out = {'finance': fr, 'science': sr}
    safe_name = EXTRACTOR_MODEL.replace(':', '_').replace('.', '_')
    output_file = os.path.join(OUTPUT_DIR, 'phase1_v2_' + safe_name + '.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print('\n' + SEP)
    print('EXTRACTION COMPLETE: ' + EXTRACTOR_MODEL)
    print('Saved to: ' + output_file)
    print('  Finance: ' + str(fr['original_len']) + '->' + str(fr['compact_len']) + ' (' + str(fr['compact_ratio']) + ':1) ' + str(fr['elapsed']) + 's')
    print('  Science: ' + str(sr['original_len']) + '->' + str(sr['compact_len']) + ' (' + str(sr['compact_ratio']) + ':1) ' + str(sr['elapsed']) + 's')
    print('  Finance entities: ' + str(fr['entity_count']) + ' (A=' + str(fr['ents_with_attrs']) + ' R=' + str(fr['ents_with_rels']) + ' empty_A=' + str(fr['empty_attrs']) + ')')
    print('  Science entities: ' + str(sr['entity_count']) + ' (A=' + str(sr['ents_with_rels']) + ' R=' + str(sr['ents_with_rels']) + ' empty_A=' + str(sr['empty_attrs']) + ')')
    print('  VRAM peak: fin=' + str(fr['vram_peak_mb']) + 'MB sci=' + str(sr['vram_peak_mb']) + 'MB')
    print(SEP)
