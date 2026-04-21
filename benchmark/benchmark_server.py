#!/usr/bin/env python3
import subprocess
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import gc
import evaluate
from openai import OpenAI
from fastapi import HTTPException
from gptq_speed import gptqmodel_speed

class RoleContent(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[RoleContent]
    max_tokens: int
    temperature: float
    repetition_penalty: float

class ChatCompletionResponse(BaseModel):
    chat: RoleContent
    prompt_per_second: float
    predicted_per_second: float
    
class PerplexityRequest(BaseModel):
    texts: List[str]
    amount: int

class BenchLLMServer:
    #      client                                        server
    #setup client                  ->{meta} fastapi               -> llm
    #query client {chat,metrics}<- ->{chat} fastapi {chat}<- ->{chat}llm
    
    app=FastAPI() # one server instance for all benches
    def __init__(self,serv_path,ppl_path,fastapi_port="8080",llm_port="8081"):
        self.executable_server_path=serv_path
        self.server_type=None # "llama.cpp" or "gptqmodel"
        self.executable_ppl_path=ppl_path
        self.model_id=None # local path or hf link
        self.fastapi_port=fastapi_port # exposed for external requests
        self.llm_port=llm_port # only for internal requests
        self.llm_openai_client=None
        self.llm_pipe=None # automatic server setup
        self.gptqmodel=None # need for measuring speed
        
        self.results={
                        'accuracy':[],
                        'ppl':[],
                        'flips':[],
                        'prompt_per_second':[],
                        'predicted_per_second':[]
                    }
    
    def start_llm_server(self,request):
        self.model_id=request.get("model_id")
        print(self.model_id)
        if request.get("server_type")=="llama.cpp":
            # llamacpp gguf bench
            self.server_type="llama.cpp"
            self.llm_pipe=subprocess.Popen([self.executable_server_path, '-m', self.model_id, '--port',self.llm_port], stdout=subprocess.PIPE)

        elif request.get("server_type")=="gptqmodel":
            # gptqmodel safetensors bench
            self.server_type="gptqmodel"
            from gptqmodel import GPTQModel, BACKEND
            self.gptqmodel=GPTQModel.load(self.model_id,device="cpu")#,device="cuda:0", backend=BACKEND.TRITON)
            # need to run separate process because current is blocked by fastapi uvicorn loop
            self.llm_pipe = subprocess.Popen(
                [
                    "python", "-c",
                    "from gptqmodel import GPTQModel, BACKEND; "
                    f"model = GPTQModel.load('{self.model_id}', device='cuda:0',backend=BACKEND.TRITON); "
                    f"model.serve(host='0.0.0.0', port={self.llm_port}, async_mode=True)"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
    
    def stop_llm_server(self):
        if self.server_type=="llama.cpp":
            self.model_id=None
            self.llm_openai_client=None
            self.llm_pipe.stdout.close()
            self.llm_pipe.terminate()
            self.llm_pipe=None
            self.server_type=None
        elif self.server_type=="gptqmodel":
            self.model_id=None
            self.llm_openai_client=None
            self.llm_pipe.stdout.close()
            self.llm_pipe.terminate()
            self.model_id=None
            self.server_type=None
            self.gptqmodel=None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    
    def setup_openai_client(self):
        self.llm_openai_client=OpenAI(base_url=f"http://localhost:{self.llm_port}/v1",api_key="")

    @app.post("/server_setup")
    async def server_setup(request: Request):
        request=await request.json()
        if request.get("status")=="start":
            benchllmserver.start_llm_server(request)
        elif request.get("status")=="stop":
            benchllmserver.stop_llm_server()
        else:
            raise Exception("error server setup")

    @app.get("/bench_v1_chat_completions",response_model=ChatCompletionResponse)
    async def bench_v1_chat_completions(request: ChatCompletionRequest):
        if benchllmserver.llm_openai_client==None:
            benchllmserver.setup_openai_client()
        response = benchllmserver.llm_openai_client.chat.completions.create(
            model=benchllmserver.model_id,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        clean_response={}
        clean_response['chat']={'content':response.choices[0].message.content if response.choices[0].message else "",
                                'role':response.choices[0].message.role if response.choices[0].message else ""}
        if benchllmserver.server_type=="llama.cpp":
            clean_response['prompt_per_second']=response.timings['prompt_per_second']
            clean_response['predicted_per_second']=response.timings['predicted_per_second']
        # manually bench speed for gptqmodel
        elif benchllmserver.server_type=="gptqmodel":
            message=f"{request.messages[-2].content} {request.messages[-1].content}"
            response_speed=gptqmodel_speed(benchllmserver.gptqmodel,message)
            clean_response['prompt_per_second']=response_speed.prompt_per_second[0]
            clean_response['predicted_per_second']=response_speed.predicted_per_second[0]
        print(clean_response)
        return clean_response
    
    @app.post("/bench_perplexity")
    async def bench_perplexity(request: PerplexityRequest):
        if benchllmserver.server_type == "llama.cpp":
            # Write texts to temp file
            temp_file = "./wikitext.txt"
            with open(temp_file, "w", encoding='utf-8') as f:
                f.writelines(request.texts)
            
            # Run perplexity calculation
            ppl_llm_process = subprocess.Popen(
                [benchllmserver.executable_ppl_path, '-m', benchllmserver.model_id, 
                '--chunks', str(request.amount), '-f', temp_file], 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            stdout, stderr = ppl_llm_process.communicate()
            
            # Parse results
            ppl_res = stdout[stdout.find('minutes'):stdout.rfind('estimate')]
            results = []
            for num in ppl_res.split(',')[:-1]:
                num = num[3:]
                results.append(num)
            return results
        
        elif benchllmserver.server_type == "gptqmodel":
            perplexity = evaluate.load("perplexity", module_type="metric")
            results = perplexity.compute(predictions=request.texts, model_id=benchllmserver.model_id)
            return results['perplexities']

if __name__ =="__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Execute the model optimization llm_processline")
    
    parser.add_argument("--serv-path",default="./llama.cpp/build/bin/llama-server")
    parser.add_argument("--ppl-path", default="./llama.cpp/build/bin/llama-perplexity")
    # gptq, awq setup comes from gptqmodel library
    args = parser.parse_args()
    
    benchllmserver=BenchLLMServer(args.serv_path,args.ppl_path)
    uvicorn.run(benchllmserver.app, host="0.0.0.0", port=8080)