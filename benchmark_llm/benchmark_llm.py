from datasets import load_dataset
import subprocess
import requests
import json

class BenchLLM:
    def __init__(self,serv_path,ppl_path,model_path):
        self.model_path=model_path
        self.executable_server_path=serv_path
        self.executable_ppl_path=ppl_path
        self.dataset=load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        self.pipe=None
        self.results={'token/sec':[],
                      'accuracy':[],
                      'ppl':[],
                      'flips':[],
                      }
    def start_server(self):
        # pipe have response, statistics
        self.pipe=subprocess.Popen([self.executable_server_path, '-m', self.model_path], stdout=subprocess.PIPE)
    def stop_server(self):
        self.pipe.stdout.close()
    def bench(self):
        headers = {"Content-Type": "application/json"}
        amount=200
        for i in range(amount):
            row=self.dataset['train'][i]
            # request have question and context titles
            data={ "prompt": row['question']+"".join(row['context']['title']),"n_predict": 256}
            response=requests.post('http://localhost:8080/completion',headers=headers,json=data).json()
            # save stats
            self.results['token/sec'].append(response['timings']['predicted_per_second'])
            self.results['accuracy'].append(any(word in response['content'].split() for word in row['answer'].split()))
        self.stop_server()
        
        ppl_pipe=subprocess.Popen(
            [self.executable_ppl_path, '-m', self.model_path, '--chunks', str(amount), '-f', '../../llama.cpp/wikitext-2-raw/wiki.train.raw'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout,stderr=ppl_pipe.communicate()
        ppl_res=stdout[stdout.find('minutes'):stdout.rfind('estimate')]
        for num in ppl_res.split(',')[:-1]:
            num=num[3:]
            self.results['ppl'].append(num)
    def save_to_file(self,filename):
        with open('name.json','w') as json_file:
            json.dump(self.results,json_file,indent=2)