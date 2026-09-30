import json, urllib.request, time

def post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request('http://127.0.0.1:8778/v1/chat/completions', data=data, headers={'Content-Type':'application/json'})
    t0 = time.time()
    resp = urllib.request.urlopen(req)
    body = resp.read().decode()
    return time.time()-t0, body

p1 = {'model':'gpt5.6-sol-flash','messages':[{'role':'user','content':'hi one sentence'}],'max_tokens':100}
dt, body = post(p1)
print('simple_elapsed', dt)
print(body)