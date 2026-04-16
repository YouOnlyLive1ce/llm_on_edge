from datasets import load_dataset
import requests
import json
import time

class BenchLLMClient:
    def __init__(self,server_url):
        self.model_id=None
        self.server_type=None
        self.dataset=load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        self.pipe=None # automatic server setup
        self.results={
                        'accuracy':[],
                        'ppl':[],
                        'flips':[],
                        'prompt_per_second':[],
                        'predicted_per_second':[]
                    }
        self.server_url=server_url
    
    def start_server(self,model_id):
        self.server_type="llama.cpp" if "gguf" in self.model_id else "gptqmodel"
        headers = {"Content-Type": "application/json"}
        data={"model_id":model_id,"server_type":self.server_type,"status":"start"}
        print(requests.post(f'{self.server_url}/server_setup',headers=headers,json=data).json())
        # wait for response setup?
    
    def stop_server(self):
        headers = {"Content-Type": "application/json"}
        data={"server_type":self.server_type,"status":"stop"}
        print(requests.post(f'{self.server_url}/server_setup',headers=headers,json=data).json())
        
    def bench_accuracy(self):
        headers = {"Content-Type": "application/json"}
        amount = 200

        for i in range(amount):
            print(i)
            row = self.dataset['validation'][i]
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": "Hello"
                },
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?"
                },
                {
                    "role": "user",
                    "content": f"Question context: {" ".join(row['context']['title'])}. \
                                Answer my question: {row['question']}. Correct answer:"
                },
            ]
            
            data = {
                "messages": conversation,
                "max_tokens": 64,
                "temperature": 0.05,
                "repetition_penalty":1.1,
            }
            
            response = requests.post(f"{self.server_url}/bench_v1_chat_completions", 
                                    headers=headers, 
                                    json=data).json()
            
            # save stats
            print(response)
            answer = response['choices'][0]['message']['content'].strip()
            self.results['accuracy'].append(any(word in answer.split() for word in row['answer'].split()))
            self.results['prompt_per_second'].append(response['timings']['prompt_per_second'])
            self.results['predicted_per_second'].append(response['timings']['predicted_per_second'])                
    
    def bench_perplexity(self):
        # Setup data
        with open("./wikitext-2-raw/wiki.test.raw") as f:
            raw_texts=f.readlines()
        texts=[]
        for i in range(len(raw_texts)):
            if len(raw_texts[i])>40:
                texts.append(raw_texts[i][:8000])
        headers = {"Content-Type": "application/json"}
        data = {
            "server_type":self.server_type,
            "texts":texts
        }
        response = requests.post(f"{self.server_url}/bench_perplexity", 
                                headers=headers, 
                                json=data).json()
        self.results['ppl']=response
    
    def save_to_file(self,filename):
        import os
        os.makedirs(os.path.dirname(f'./{filename}.json'), exist_ok=True)
        with open(f'./{filename}.json','w') as json_file:
            json.dump(self.results,json_file,indent=2)
        self.results={'accuracy':[],'ppl':[],'flips':[],'prompt_per_second':[],'predicted_per_second':[]}
    
    def compare_results(self, path_to_jsons, base_model_file_name):
        import os
        import pandas as pd
        import numpy as np
        import scipy.stats

        json_files = [pos_json for pos_json in os.listdir(path_to_jsons) if pos_json.endswith('.json')]
        print(f"json_files: {json_files}")
        dfs = []
        for file in json_files:
            df = pd.read_json(path_to_jsons + "/" + file, orient='index')
            df = df.transpose()
            if df['ppl'].dtype==object: #todo remove if
                corrected = df['ppl'].str.extract(r'\](\d+\.?\d*)')
                corrected = corrected[0]  # Extract from DataFrame to Series
                mask = corrected.notna()
                df.loc[mask, 'ppl'] = corrected[mask]
                df['ppl'] = pd.to_numeric(df['ppl'], errors='coerce')
            df[~np.isfinite(df['ppl'])]=df[np.isfinite(df['ppl'])].mean() 
            df['accuracy'] = pd.to_numeric(df['accuracy'])
            df['ppl'] = pd.to_numeric(df['ppl'])
            dfs.append(df)
            print(df['accuracy'].notna().sum(),df['ppl'].notna().sum())

        # Find index of the base model
        base_model_idx = None
        for i, fname in enumerate(json_files):
            if fname == base_model_file_name:
                base_model_idx = i
                break
        if base_model_idx is None:
            raise ValueError(f"Base model file '{base_model_file_name}' not found in json_files")

        # Flip flags: compare each model to the base model
        for i in range(len(dfs)):
            if i == base_model_idx:
                # Base model has no flips relative to itself
                dfs[i]['flips'] = False
            else:
                dfs[i]['flips'] = (dfs[i]['accuracy'] == 0.0) & (dfs[base_model_idx]['accuracy'] == 1.0)

        pd.set_option('display.float_format', '{:0.20f}'.format)
        result_df = []

        for i in range(len(json_files)):
            df = dfs[i]

            # filter rows which fail to generate tokens
            fail_mask = df['token/sec'] == 1000000.0
            fail_count = fail_mask.sum()
            df_good = df[~fail_mask]

            def mean_std(series):
                if series is None or len(series) == 0:
                    return "NaN"
                return f"{series.mean():.2f}+-{series.std():.2f}"

            # do not count 0 generated tokens with 1000000 tok/sec
            ppl_str = mean_std(df['ppl'])
            prefill_str = mean_std(df_good['prompt_per_second'])
            decode_str = mean_std(df_good['predicted_per_second'])
            # do count 0 generated tokens as fail
            accuracy_mean = df['accuracy'].sum()/df['accuracy'].notna().sum()
            flips_mean = df['flips'].sum()/df['accuracy'].notna().sum()

            # McNemar's test p-value relative to base model
            if i == base_model_idx:
                p_value = np.nan
            else:
                base_acc = dfs[base_model_idx][~fail_mask]['accuracy']
                curr_acc = df['accuracy']
                # Contingency table:
                # a: both correct
                # b: base correct, current incorrect (flips)
                # c: base incorrect, current correct
                # d: both incorrect
                b = ((base_acc == 1) & (curr_acc == 0)).sum()
                c = ((base_acc == 0) & (curr_acc == 1)).sum()
                print("correct->incorrect",b)
                print("incorrect->correct",c)
                if b + c == 0:
                    p_value = np.nan
                else:
                    chi2 = (b - c) ** 2 / (b + c)
                    p_value = 1 - scipy.stats.chi2.cdf(chi2, 1)

            result_df.append({
                "method": json_files[i],
                "accuracy": f"{accuracy_mean:.2f}",
                "ppl": ppl_str,
                "prefill": prefill_str,
                "decode": decode_str,
                # "memory": memory_str,
                "flips": f"{flips_mean:.2f}",
                "fail": f"{fail_count/df['accuracy'].size:.2f}",
                "p_value": f"{p_value:.2f},n={df['accuracy'].notna().sum()}"
            })

        result_df = pd.DataFrame(result_df)
        return result_df
    
    def calc_corr(self, result_df):
        import pandas as pd
        import numpy as np
        
        tmp_df = result_df.copy(deep=True)
        
        def extract_mean(value):
            if pd.isna(value) or value == 'NaN':
                return np.nan
            try:
                if isinstance(value, str) and '+-' in value:
                    return float(value.split('+-')[0])
                else:
                    return float(value)
            except (ValueError, AttributeError):
                return np.nan
        
        # Extract numeric values from string columns
        for col in ['decode', 'ppl', 'token/sec', 'prefill', 'memory']:
            if col in tmp_df.columns:
                tmp_df[col] = tmp_df[col].apply(extract_mean)
        
        # Drop non-numeric columns
        cols_to_drop = ['method'] if 'method' in tmp_df.columns else []
        if 'p_value' in tmp_df.columns:
            cols_to_drop.append('p_value')
        if 'flips' in tmp_df.columns:
            cols_to_drop.append('flips')
        if 'fails' in tmp_df.columns:
            cols_to_drop.append('fails')
        
        tmp_df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
        tmp_df = tmp_df.apply(pd.to_numeric, errors='coerce')
        
        return tmp_df.corr()

if __name__ =="__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Execute the model optimization pipeline")
    parser.add_argument("--outs-path", help="where to save benchmark results")
    parser.add_argument("--server-url",default="http://localhost:8080")
    args = parser.parse_args()
    
    # parse all models
    models=[]
    client=BenchLLMClient(args.server_url)
    for model in models:
        client.start_server(model)
        time.sleep(300) # wait for model to load
        client.bench_accuracy()
        client.bench_perplexity()
        client.save_to_file(f'./{args.out_file}')
        client.stop_server()