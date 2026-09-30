import json, urllib.request, time, sys
key = 'REPLACE_WITH_YOUR_API_KEY'
port = sys.argv[1] if len(sys.argv) > 1 else '8780'
content = sys.argv[2] if len(sys.argv) > 2 else '请详细介绍一下你自己，包括能力、特点和一个具体例子，写200字'
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 300
payload = {'model':'gpt5.6 sol','messages':[{'role':'user','content':content}],'max_tokens':max_tokens,'stream':True,'temperature':0.7}
data = json.dumps(payload).encode()
req = urllib.request.Request('http://127.0.0.1:%s/v1/chat/completions'%port, data=data, headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
t0=time.time(); ttft=None; n_tok=0; n_reason=0
resp = urllib.request.urlopen(req, timeout=600)
for raw in resp:
    line=raw.decode('utf-8','replace').strip()
    if not line.startswith('data:'): continue
    j=line[5:].strip()
    if j=='[DONE]': break
    try: obj=json.loads(j)
    except: continue
    ch=obj.get('choices',[])
    if not ch: continue
    d=ch[0].get('delta',{}) or {}
    c=d.get('content') or ''; r=d.get('reasoning_content') or ''
    if c or r:
        now=time.time()
        if ttft is None: ttft=now-t0
        if c: n_tok+=1
        if r: n_reason+=1
t1=time.time(); total=t1-t0; gen=t1-ttft if ttft else 0
print('port:',port,'| content_tok:',n_tok,'reason_tok:',n_reason)
print('TTFT_s: %.3f'%(ttft if ttft else -1))
print('total_s: %.3f'%total)
print('decode_tok_s(content): %.2f'%(n_tok/gen if gen>0 else 0))
print('decode_tok_s(all): %.2f'%((n_tok+n_reason)/gen if gen>0 else 0))