import json, urllib.request, time, sys

key = 'REPLACE_WITH_YOUR_API_KEY'
model = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5.6-sol'
content = sys.argv[2] if len(sys.argv) > 2 else '请详细介绍一下你自己，包括你的能力、特点和一个例子，写300字左右'
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 500

payload = {'model': model, 'messages': [{'role':'user','content':content}], 'max_tokens': max_tokens, 'stream': True, 'temperature': 0.7}
data = json.dumps(payload).encode()
req = urllib.request.Request('http://127.0.0.1:8778/v1/chat/completions', data=data, headers={'Content-Type':'application/json','Authorization':'Bearer '+key})

t0 = time.time()
ttft = None
n_tok = 0
n_reason = 0
last = time.time()
resp = urllib.request.urlopen(req, timeout=600)
for raw in resp:
    line = raw.decode('utf-8','replace').strip()
    if not line.startswith('data:'):
        continue
    j = line[len('data:'):].strip()
    if j == '[DONE]':
        break
    try:
        obj = json.loads(j)
    except Exception:
        continue
    choices = obj.get('choices', [])
    if not choices:
        continue
    delta = choices[0].get('delta', {}) or {}
    c = delta.get('content') or ''
    r = delta.get('reasoning_content') or ''
    if c or r:
        now = time.time()
        if ttft is None:
            ttft = now - t0
        if c:
            n_tok += 1
        if r:
            n_reason += 1
        last = now

t1 = time.time()
total = t1 - t0
gen = t1 - ttft if ttft is not None else 0
all_tok = n_tok + n_reason
print('model:', model)
print('content_tokens:', n_tok, ' reason_tokens:', n_reason, ' all:', all_tok)
print('TTFT_s: %.3f' % (ttft if ttft is not None else -1))
print('total_s: %.3f' % total)
print('decode_tok_s (all): %.2f' % (all_tok/gen if gen > 0 else 0))
print('decode_tok_s (content): %.2f' % (n_tok/gen if gen > 0 else 0))
print('e2e_tok_s: %.2f' % (all_tok/total if total > 0 else 0))