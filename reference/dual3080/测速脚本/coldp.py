import json, urllib.request, time, sys, random, string

key = 'REPLACE_WITH_YOUR_API_KEY'
port = sys.argv[1] if len(sys.argv) > 1 else '8778'
target_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 16384

nonce = ''.join(random.choices(string.ascii_letters+string.digits, k=16))
para = '量子计算利用量子比特的叠加和纠缠态进行并行计算，在分解大整数、搜索无结构数据库等特定问题上展现出经典计算机难以企及的优势。通过精心设计的算法如Shor算法与Grover算法，以及量子纠错和量子门操作的配合，量子计算机能够显著降低某些计算任务的复杂度，从而改变我们对信息处理极限的理解。这是一段用于测试长上下文预填充性能的中文文本，用来模拟真实文档阅读场景。'
# 约 1 中文/字 接近 1 token；先估算：每段 ~120 中文字 -> ~130 token
# 逐段累加直到达到目标（用字符数粗估，再靠服务端返回的 prompt_tokens 校正）
content = '【nonce:%s】\n' % nonce
chars_per_para = len(para)
est_tok_per_para = 130
n = max(1, (target_tok - len(content)) // est_tok_per_para)
content += para * n
max_tokens = 1  # 只要 TTFT，尽量少生成
payload = {'model':'gpt-5.6-sol','messages':[{'role':'user','content':content}],'max_tokens':max_tokens,'stream':False,'temperature':0.7}
data = json.dumps(payload).encode()
req = urllib.request.Request('http://127.0.0.1:%s/v1/chat/completions'%port, data=data, headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
t0=time.time()
resp = urllib.request.urlopen(req, timeout=600)
t1=time.time()
obj = json.loads(resp.read().decode('utf-8','replace'))
usage = obj.get('usage',{})
pt = usage.get('prompt_tokens', -1)
ct = usage.get('completion_tokens', 0)
print('port:',port,'| target_tok_est:',target_tok)
print('actual_prompt_tokens:',pt)
print('TTFT_s: %.3f'%(t1-t0))
print('prompt_tokens_per_s: %.1f'%(pt/(t1-t0) if (t1-t0)>0 else 0))