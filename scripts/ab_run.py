import argparse, json, os, time, urllib.request, urllib.error

def call(endpoint, key, body, timeout):
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw), time.time() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, "HTTP %s: %s" % (e.code, e.read()[:400])
    except Exception as e:
        return None, time.time() - t0, "%s: %s" % (type(e).__name__, e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True); ap.add_argument("--endpoint", required=True)
    ap.add_argument("--key", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=2700)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    tests = [json.loads(l) for l in open(a.tests, encoding="utf-8") if l.strip()]
    if a.only:
        want = set(a.only.split(","))
        tests = [t for t in tests if t["id"] in want]
    res = []
    for t in tests:
        body = {"model": a.model, "messages": t["messages"], "max_tokens": t["max_tokens"],
                "temperature": a.temperature, "top_p": 0.95, "stream": False}
        if t.get("tools"): body["tools"] = t["tools"]
        resp, dt, err = call(a.endpoint, a.key, body, a.timeout)
        rec = {"id": t["id"], "category": t["category"], "wall_s": round(dt, 2), "error": err}
        if resp:
            ch = (resp.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            rec["content"] = msg.get("content")
            rec["reasoning"] = msg.get("reasoning_content")
            rec["tool_calls"] = msg.get("tool_calls")
            rec["finish"] = ch.get("finish_reason")
            rec["usage"] = resp.get("usage")
        res.append(rec)
        print("%-16s %-8s %7.1fs %s" % (t["id"], t["category"], dt, err or ("len=%d" % len(rec.get("content") or ""))), flush=True)
        os.makedirs(a.out, exist_ok=True)
        json.dump({"endpoint": a.endpoint, "model": a.model, "temperature": a.temperature, "results": res},
                  open(os.path.join(a.out, "results_%s.json" % a.tag), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("done, wrote results_%s.json (%d tests)" % (a.tag, len(res)))

if __name__ == "__main__":
    main()
