#!/usr/bin/env python3
'''新机器部署完的自检脚本：模型信息 / key 校验 / 普通对话 / 长文找数字 / TTFT 与出字速度。

用法：
  python verify.py --endpoint http://127.0.0.1:8080 --key <APIKEY> --model qwen3.8-flash-next-iq3_s
  python verify.py --endpoint http://127.0.0.1:8778 --key <APIKEY> --model Qwen3.8-27B --long-tokens 98000
'''
import argparse, json, time, urllib.request, urllib.error, random, string, sys

def _req(url, data=None, key=None, timeout=3600):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    body = json.dumps(data, ensure_ascii=False).encode('utf-8') if data is not None else None
    return urllib.request.Request(url, data=body, headers=headers, method='POST' if data is not None else 'GET')

def check_models(ep, key):
    url = ep.rstrip('/') + '/v1/models'
    d = json.loads(urllib.request.urlopen(_req(url, key=key), timeout=30).read())
    m = d['data'][0]
    ident = m.get('id')
    ctx = (m.get('meta') or {}).get('n_ctx') or m.get('max_model_len')
    print('[1] /v1/models OK  model=%s  context=%s' % (ident, ctx))
    return ident, ctx

def check_no_key(ep):
    url = ep.rstrip('/') + '/v1/models'
    try:
        urllib.request.urlopen(_req(url), timeout=20)
        print('[2] WARNING: 不带 key 也能访问，key 校验没生效')
        return False
    except urllib.error.HTTPError as e:
        print('[2] 不带 key 被拒，HTTP %s（期望 401）' % e.code)
        return e.code == 401
    except Exception as e:
        print('[2] 不带 key 的请求异常：%s' % e)
        return False

def chat(ep, key, body, stream=False):
    url = ep.rstrip('/') + '/v1/chat/completions'
    b = dict(body)
    if stream:
        b['stream'] = True
        b['stream_options'] = {'include_usage': True}
    r = urllib.request.urlopen(_req(url, b, key), timeout=3600)
    if not stream:
        d = json.loads(r.read())
        ch = d['choices'][0]
        return {'text': ch['message'].get('content') or '', 'reasoning': ch['message'].get('reasoning_content'),
                'finish': ch.get('finish_reason'), 'usage': d.get('usage'), 'ttft': None}
    ttft = None
    usage = {}
    parts = []
    t0 = time.time()
    for raw in r:
        s = raw.decode('utf-8', 'ignore').strip()
        if not s.startswith('data:'):
            continue
        p = s[5:].strip()
        if p == '[DONE]':
            break
        try:
            d = json.loads(p)
        except Exception:
            continue
        delta = (d.get('choices') or [{}])[0].get('delta') or {}
        if delta.get('content'):
            if ttft is None:
                ttft = time.time() - t0
            parts.append(delta['content'])
        if d.get('usage'):
            usage = d['usage']
    return {'text': ''.join(parts), 'finish': None, 'usage': usage,
            'ttft': ttft, 'total': time.time() - t0}

def build_long_doc(tokens, fact_id, fact_value):
    words = ['audit', 'register', 'vendor', 'migration', 'staffing', 'quarterly', 'throughput',
             'budgets', 'contracts', 'minutes', 'review', 'archived', 'lead', 'risk']
    lines = []
    n = max(1, tokens // 25)
    for i in range(n):
        lines.append('[%05d] %s' % (i, ' '.join(random.choice(words) for _ in range(20))))
    lines[n // 2] = '[%05d] %s 的最终结算金额是 %s 元。' % (n // 2, fact_id, fact_value)
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--long-tokens', type=int, default=98000)
    ap.add_argument('--think', action='store_true', help='测试时保留思考（默认关闭）')
    a = ap.parse_args()
    nothink = {} if a.think else {'chat_template_kwargs': {'enable_thinking': False}}

    check_models(a.endpoint, a.key)
    check_no_key(a.endpoint)

    body = {'model': a.model, 'messages': [{'role': 'user', 'content': '用一句话介绍你自己'}],
            'max_tokens': 200, 'temperature': 0.3}
    body.update(nothink)
    t0 = time.time()
    r = chat(a.endpoint, a.key, body)
    ct = (r['usage'] or {}).get('completion_tokens')
    print('[3] 普通对话 OK  %.2fs  out_tokens=%s  finish=%s' % (time.time() - t0, ct, r['finish']))
    print('    reply: %s' % (r['text'][:80].replace('\n', ' ')))

    fact = '837462.19'
    doc = build_long_doc(a.long_tokens, 'KR-4471', fact)
    q = doc + '\n\n以上是全部文档。问题：文档中编号 KR-4471 的最终结算金额是多少元？只回答数字。'
    lb = {'model': a.model, 'messages': [{'role': 'user', 'content': q}], 'max_tokens': 24, 'temperature': 0.0}
    lb.update(nothink)
    print('[4] 长文测试（首次，冷缓存）...')
    r1 = chat(a.endpoint, a.key, lb, stream=True)
    pt = (r1['usage'] or {}).get('prompt_tokens')
    ct = (r1['usage'] or {}).get('completion_tokens') or 0
    gen = (r1['total'] - r1['ttft']) if r1['ttft'] else r1['total']
    print('    prompt_tokens=%s  TTFT=%.2fs  total=%.2fs  decode=%.1f tok/s' % (pt, r1['ttft'] or -1, r1['total'], ct / gen if gen > 0 else 0))
    print('    answer=%r  期望包含 %s  -> %s' % (r1['text'].strip()[:40], fact, 'PASS' if fact in r1['text'] else 'FAIL'))
    r2 = chat(a.endpoint, a.key, lb, stream=True)
    print('[5] 长文重复（应命中缓存）TTFT=%.2fs  total=%.2fs' % (r2['ttft'] or -1, r2['total']))

if __name__ == '__main__':
    main()
