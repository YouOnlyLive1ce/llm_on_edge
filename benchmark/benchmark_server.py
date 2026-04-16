import subprocess
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, float
from gptqmodel import GPTQModel
import json
import gc
import evaluate
import requests
import openai
import time

class RoleContent(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[RoleContent]
    max_tokens: int
    temperature: float
    repetition_penalty: float

class ChatCompletionResponse(BaseModel):
    chat: List[RoleContent]
    prompt_per_second: float
    predicted_per_second: float

class BenchLLMServer:
    #      client                                        server
    #setup client                  ->{meta} fastapi               -> llm
    #query client {chat,metrics}<- ->{chat} fastapi {chat}<- ->{chat}llm
    app=FastAPI() # one server instance for all benches
    def __init__(self,serv_path,ppl_path,model_path,fastapi_port="8080",llm_port="8081"):
        self.executable_server_path=serv_path
        self.server_type=None # "llama.cpp" or "gptqmodel"
        self.executable_ppl_path=ppl_path
        self.model_path=model_path # local path or hf link
        self.fastapi_port=fastapi_port
        self.llm_port=llm_port
        self.llm_openai_client=None
        self.pipe=None # automatic server setup
        self.results={
                        'accuracy':[],
                        'ppl':[],
                        'flips':[],
                        'prompt_per_second':[],
                        'predicted_per_second':[]
                    }
    
    @app.post("/server_setup")
    async def server_setup(self,request: Request):
        request=await request.json()
        if request.get("status")=="start":
            self.start_llm_server(request)
        elif request.get("status")=="stop":
            self.stop_llm_server()
        else:
            raise Exception("error server setup")
    
    def start_llm_server(self,request):
        if request.get("server_type")=="llama.cpp":
            # llamacpp gguf bench
            self.server_type="llama.cpp"
            self.pipe=subprocess.Popen([self.executable_server_path, '-m', self.model_path, '--port',self.llm_port], stdout=subprocess.PIPE)
            
        elif request.get("server_type")=="gptqmodel":
            # gptqmodel safetensors bench
            self.server_type="gptqmodel"
            self.model=GPTQModel.load(self.model_path,device="cpu")
            self.model.serve(host="0.0.0.0",port=self.llm_port,async_mode=True)
    
    def stop_llm_server(self):
        if self.server_type=="llama.cpp":
            self.pipe.stdout.close()
            self.pipe.terminate()
            self.pipe=None
            self.server_type=None
        elif self.server_type=="gptqmodel":
            self.model.serve_shutdown()
            self.model=None
            self.model_path=None
            self.server_type=None
        gc.collect()
    
    def setup_openai_client(self):
        self.llm_openai_client=openai.Client(base_url=f"{self.server_url}/v1",api_key="None")
    
    @app.get("/bench_v1_chat_completions",response_model=ChatCompletionResponse)
    async def bench_v1_chat_completions(self,request: ChatCompletionRequest):
        # start_time=time.time()
        response = self.llm_openai_client.chat.completions.create(
            model=self.MODEL_ID,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty
        )
        # prompt_and_predicted_time=time.time()-start_time
        # append prompt_per_second and predicted_per_second
        # if self.server_type=="gptqmodel":
        #     response['prompt_per_second']=
        return response
    
    @app.get("/bench_perplexity",response_model=List[float])
    async def bench_perplexity(self,request:Request):
        if request.get("server_type")=="llama.cpp":
            with open("./wikitext.txt","w") as f:
                f.write(request.get("texts"))
            ppl_pipe=subprocess.Popen(
                [self.executable_ppl_path, '-m', self.model_path, '--chunks', request.get("amount"), '-f', './wikitext.txt'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            stdout,stderr=ppl_pipe.communicate()
            ppl_res=stdout[stdout.find('minutes'):stdout.rfind('estimate')]
            results=[]
            for num in ppl_res.split(',')[:-1]:
                num=num[3:]
                results['ppl'].append(float(num))
            return results
        elif request.get("server_type")=="gptqmodel":  
            perplexity = evaluate.load("perplexity", module_type="metric")
            results = perplexity.compute(predictions=request.get("texts"), model_id=self.model_id)
            return results

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Execute the model optimization pipeline")
    
    parser.add_argument("--serv-path",default="./llama.cpp/build/bin/llama-server")
    parser.add_argument("--ppl-path", default="./llama.cpp/build/bin/llama-perplexity")
    parser.add_argument("--model-path", help="Path to gguf file or folder")
    args = parser.parse_args()
    
    benchllm=BenchLLMServer(args.serv_path,args.ppl_path,args.model_path)