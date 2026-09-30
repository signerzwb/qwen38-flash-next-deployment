import json, urllib.request, time
key='REPLACE_WITH_YOUR_API_KEY'
# /v1/models
req=urllib.request.Request('http://127.0.0.1:8778/v1/models',headers={'Authorization':'Bearer '+key})
print('=== /v1/models ===')
try:
    r=urllib.request.urlopen(req,timeout=15)
    print(r.status, r.read().decode()[:200])
except Exception as e: print('ERR',e)
# chat
print('=== chat ===')
payload={'model':'gpt5.6 sol','messages':[{'role':'user','content':'你好'}],'max_tokens':60,'stream':False}
req=urllib.request.Request('http://127.0.0.1:8778/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
try:
    r=urllib.request.urlopen(req,timeout=120)
    obj=json.loads(r.read().decode())
    print('status',r.status,'| msg:',obj['choices'][0]['message']['content'][:80])
except Exception as e: print('ERR',e)