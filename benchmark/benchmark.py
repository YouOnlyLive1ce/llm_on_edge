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
        self.pipe=subprocess.Popen([self.executable_server_path, '-m', self.model_path, '--port','8081'], stdout=subprocess.PIPE)
    def stop_server(self):
        self.pipe.stdout.close()
        self.pipe.terminate()
    def bench(self):
        headers = {"Content-Type": "application/json"}
        amount=2
        for i in range(amount):
            print(i)
            row=self.dataset['train'][i]
            # request have question and context titles
            data={ "prompt": row['question']+"".join(row['context']['title']),"n_predict": 256}
            response=requests.post('http://localhost:8081/completion',headers=headers,json=data).json()
            # save stats
            self.results['token/sec'].append(response['timings']['predicted_per_second'])
            self.results['accuracy'].append(any(word in response['content'].split() for word in row['answer'].split()))
        self.stop_server()
        
        # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        ppl_pipe=subprocess.Popen(
            [self.executable_ppl_path, '-m', self.model_path, '--chunks', str(amount), '-f', './wikitext-2-raw/wiki.train.raw'], 
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
        with open(f'./benchmark/benchmark_results/{filename}.json','w') as json_file:
            json.dump(self.results,json_file,indent=2)
    
    def compare_results(self, path_to_jsons):
        import os
        import pandas as pd
        import numpy as np

        path_to_jsons = '.'
        # last df has highest bitwidth and hence used to compare with
        json_files = [pos_json for pos_json in os.listdir(path_to_jsons) if pos_json.endswith('.json')]
        dfs=[]
        for file in json_files:
            df=pd.read_json(file, orient='index')
            df = df.transpose()
            corrected=df['ppl'].str.extract(r'\](\d+\.?\d*)')
            corrected = corrected[0]  # Extract from DataFrame to Series
            mask = corrected.isna()
            corrected[mask] = df['ppl'][mask]
            df['ppl']=corrected
            df['accuracy']=pd.to_numeric(df['accuracy'])
            df['ppl']=pd.to_numeric(df['ppl'])
            dfs.append(df)

        # flip flags. calc relative to model with best accuracy
        best_model_idx=np.array([dfs[i]['accuracy'].mean() for i in range(len(dfs))]).argmax()
        for i in range(len(dfs)):
            if i==best_model_idx:
                continue
            dfs[i]['flips']=(dfs[i]['accuracy']==0.0) & (dfs[best_model_idx]['accuracy']==1.0)

        pd.set_option('display.float_format', '{:0.20f}'.format)
        result_df=[]
        for i in range(len(json_files)):
            result_df.append(
                {"method":json_files[i],"token/sec":f"{dfs[i]['token/sec'].mean()}+-{dfs[i]['token/sec'].std()}",
                "accuracy":f"{dfs[i]['accuracy'].mean()}","ppl":f"{dfs[i]['ppl'].mean()}+-{dfs[i]['ppl'].std()}",
                "flips":f"{dfs[i]['flips'].mean()}"}
            )
        result_df=pd.DataFrame(result_df)
        return result_df
    def calc_corr(self,result_df):
        tmp_df=result_df.copy(deep=True)
        tmp_df['token/sec']=tmp_df['token/sec'].apply(lambda x: float(x.split('+-')[0]))
        tmp_df['ppl']=tmp_df['ppl'].apply(lambda x: float(x.split('+-')[0]))
        tmp_df.drop(['method'],axis=1,inplace=True)
        return tmp_df.corr()

if __name__ =="__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Execute the model optimization pipeline")
    parser.add_argument("--model-name", default="smolm2", help="Path to config file")
    parser.add_argument("--model-path", default=".wanda/out/smollm2/unstructured/sparsegpt/smolm2-pruned.gguf", help="Path to config file")
    args = parser.parse_args()
    
    model_name=args.model_name
    model_path=args.model_path
    benchllm=BenchLLM('./bin_llama/llama-server','./bin_llama/llama-perplexity',model_path)
    time.sleep(180) # wait for previous model free and server to close
    benchllm.start_server()
    time.sleep(240) # wait for model to load
    benchllm.bench() # stop server itself
    benchllm.save_to_file(f'./{model_name}')