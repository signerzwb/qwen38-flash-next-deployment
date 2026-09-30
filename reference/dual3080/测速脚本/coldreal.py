import json, urllib.request, time, sys, random, string
key = 'REPLACE_WITH_YOUR_API_KEY'
port = sys.argv[1] if len(sys.argv) > 1 else '8780'
target_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
# 每次随机 nonce 前缀 -> 强制 radix cache 不命中 (真冷)
nonce = ''.join(random.choices(string.ascii_letters+string.digits, k=16))
random.seed(hash(nonce) & 0xffffffff)
subjects = ['量子计算','神经网络','分布式系统','数据库索引','密码学算法','图遍历','编译优化','内存管理','并发控制','模型推理']
verbs = ['通过不同的方式','在特定条件下','借助复杂机制','利用多种策略','经过若干步骤','依据不同规则','采用多种方案','在多种场景中','借助一定手段','利用不同结构']
objs = ['改变了计算复杂度','提升了吞吐效率','降低了资源开销','增强了鲁棒性','优化了延迟表现','改善了扩展能力','减少了存储消耗','提升了并行度','增强了一致性','降低了能耗']
sentences=[]
for i in range(target_tok//20 + 200):
    s = random.choice(subjects)+random.choice(verbs)+random.choice(objs)+str(i)+'号观测样本记录了该现象在长期演化中的具体特征与边界条件，用于后续科学分析。'
    sentences.append(s)
content = '【nonce:%s】\n' % nonce + '\n'.join(sentences)
payload = {'model':'gpt5.6 sol','messages':[{'role':'user','content':content}],'max_tokens':1,'stream':False,'temperature':0.7}
data = json.dumps(payload).encode()
req = urllib.request.Request('http://127.0.0.1:%s/v1/chat/completions'%port, data=data, headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
t0=time.time(); resp=urllib.request.urlopen(req, timeout=900); t1=time.time()
obj=json.loads(resp.read().decode('utf-8','replace'))
pt=obj.get('usage',{}).get('prompt_tokens',-1)
print('port:',port,'| nonce:',nonce,'| prompt_tokens:',pt)
print('TTFT_s: %.3f'%(t1-t0))
print('prompt_tokens_per_s: %.1f'%(pt/(t1-t0) if (t1-t0)>0 else 0))