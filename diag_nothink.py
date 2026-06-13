# -*- coding: utf-8 -*-
import requests, json, re
OLLAMA = "http://localhost:11434"
t = "美联储主席鲍威尔表示通胀降温，美联储决定降息25个基点。"
p = '提取文本中实体名字和关系JSON。输出:{"E":[{"n":"名","R":[...]}]}\n文本:' + t + "\n/no_think"
s = "提取器。只输出JSON。"
r = requests.post(f"{OLLAMA}/api/chat", json={"model":"qwen3.5:4b","messages":[{"role":"system","content":s},{"role":"user","content":p}],"stream":False}, timeout=120)
c = r.json().get("message",{}).get("content","")
print(f"len={len(c)}")
print(f"content=[{c[:1500]}]")
m = re.search(r"\{[\s\S]*\}", c)
print(f"json_found={bool(m)}")
if m:
    try:
        parsed = json.loads(m.group())
        print(f"entities={len(parsed.get('E',[]))}")
        for e in parsed.get('E',[])[:5]:
            print(f"  {e}")
    except:
        print("json_parse_fail")
