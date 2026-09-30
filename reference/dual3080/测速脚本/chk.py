import json
p='/root/LLM/Qwen3.8-27B-FP8/config.json'
d=json.load(open(p))
print('arch:', d.get('architectures'))
print('model_type:', d.get('model_type'))
q=d.get('quantization_config') or {}
print('quant:', json.dumps(q)[:300])
print('num_hidden_layers:', d.get('num_hidden_layers'))
print('has_mtp_weights:', any('mtp' in k for k in d))
import os
files=os.listdir('/root/LLM/Qwen3.8-27B-FP8')
print('has_vision:', any(('visual' in f.lower() or 'vision' in f.lower() or 'image' in f.lower()) for f in files))
print('layers_count:', len([f for f in files if f.startswith('layers-')]))