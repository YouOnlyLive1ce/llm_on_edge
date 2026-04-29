# llm_on_edge
Making HF model run on mobile/Raspberry

#### Usage example
```
python3 ./llm_on_edge/main.py --stages setup ; pip install requirements.txt
```
```
python3 ./llm_on_edge/main.py --stages convert_hf_to_gguf,prune,quantize --config ./llm_on_edge/config-w-02.yaml ; 
python3 ./llm_on_edge/main.py --stages convert_hf_to_gguf,prune,quantize --config ./llm_on_edge/config-w-05.yaml ; 
python3 ./llm_on_edge/main.py --stages convert_hf_to_gguf,prune,quantize --config ./llm_on_edge/config-s-02.yaml ; 
python3 ./llm_on_edge/main.py --stages convert_hf_to_gguf,prune,quantize --config ./llm_on_edge/config-s-05.yaml 
```
```
python3 ./llm_on_edge/main.py --stages benchmark.py
```