from datasets import load_dataset
import subprocess
import requests
import json
# from memory_profiler import memory_usage

class BenchLLM:
    def __init__(self,serv_path,ppl_path,model_path):
        self.executable_server_path=serv_path
        self.executable_ppl_path=ppl_path
        self.model_path=model_path
        self.dataset=load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        self.pipe=None
        self.results={'token/sec':[],
                      'accuracy':[],
                      'ppl':[],
                      'flips':[],
                      'prompt_per_second':[],
                      'predicted_per_second':[]
                      }
    def start_server(self):
        # pipe have response, statistics
        self.pipe=subprocess.Popen([self.executable_server_path, '-m', self.model_path, '--port','8081'], stdout=subprocess.PIPE)
        # self.memory_usage=memory_usage(self.pipe, interval=60,multiprocess=True)
    
    def stop_server(self):
        self.pipe.stdout.close()
        self.pipe.terminate()
        # print(self.memory_usage)
        
    def bench(self):
        headers = {"Content-Type": "application/json"}
        amount=200
        # hotpotqa is a fact knowledge checking dataset. if any of words from response is in target, count as correct
        for i in range(amount):
            print(i)
            row=self.dataset['validation'][i]
            # request have question and context titles
            data={ "prompt": "Topic: "+" ".join(row['context']['title'])+" Question: "+row['question']+ " Your answer: ","n_predict": 64}
            response=requests.post('http://localhost:8081/completion',headers=headers,json=data).json()
            # save stats
            self.results['token/sec'].append(response['timings']['predicted_per_second'])
            self.results['prompt_per_second'].append(response['timings']['prompt_per_second'])
            self.results['predicted_per_second'].append(response['timings']['predicted_per_second'])
            self.results['accuracy'].append(any(word in response['content'].split() for word in row['answer'].split()))
            print(response)
        self.stop_server()
        
        ppl_pipe=subprocess.Popen(
            [self.executable_ppl_path, '-m', self.model_path, '--chunks', str(amount), '-f', '/mnt/d/Inno/Thesis/llama.cpp/wikitext-2-raw/wiki.test.raw'], 
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
        import os
        os.makedirs(os.path.dirname(f'./{filename}.json'), exist_ok=True)
        with open(f'./{filename}.json','w') as json_file:
            json.dump(self.results,json_file,indent=2)
    
    def compare_results(self, path_to_jsons, base_model_file_name):
        import os
        import pandas as pd
        import numpy as np
        import scipy.stats  # for p-value calculation

        # last df has highest bitwidth and hence used to compare with
        json_files = [pos_json for pos_json in os.listdir(path_to_jsons) if pos_json.endswith('.json')]
        print(f"json_files: {json_files}")
        dfs = []
        for file in json_files:
            df = pd.read_json(path_to_jsons + "/" + file, orient='index')
            df = df.transpose()
            corrected = df['ppl'].str.extract(r'\](\d+\.?\d*)')
            corrected = corrected[0]  # Extract from DataFrame to Series
            mask = corrected.isna()
            corrected[mask] = df['ppl'][mask]
            df['ppl'] = corrected
            df['accuracy'] = pd.to_numeric(df['accuracy'])
            df['ppl'] = pd.to_numeric(df['ppl'])
            dfs.append(df)

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

            # Compute metrics on good rows only
            ppl_str = mean_std(df_good['ppl'])
            prefill_str = mean_std(df_good['prompt_per_second'])
            decode_str = mean_std(df_good['predicted_per_second'])
            accuracy_mean = df_good['accuracy'].mean()
            flips_mean = df_good['flips'].mean()

            # McNemar's test p-value relative to base model
            if i == base_model_idx:
                p_value = np.nan
            else:
                base_acc = dfs[base_model_idx][~fail_mask]['accuracy']
                curr_acc = df_good['accuracy']
                # Contingency table:
                # a: both correct
                # b: base correct, current incorrect (flips)
                # c: base incorrect, current correct
                # d: both incorrect
                b = ((base_acc == 1) & (curr_acc == 0)).sum()
                c = ((base_acc == 0) & (curr_acc == 1)).sum()
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
                "p_value": f"{p_value:.2f},n={df_good['accuracy'].size}"
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
    parser.add_argument("--out-file", help="where to save benchmark results")
    parser.add_argument("--serv-path",default="./llama.cpp/build/bin/llama-server")
    parser.add_argument("--ppl-path", default="./llama.cpp/build/bin/llama-perplexity")
    parser.add_argument("--model-path", help="Path to gguf file")
    args = parser.parse_args()
    
    benchllm=BenchLLM(args.serv_path,args.ppl_path,args.model_path)
    time.sleep(60) # wait for previous model free memory
    benchllm.start_server()
    time.sleep(360) # wait for model to load
    benchllm.bench() # stop server itself
    benchllm.save_to_file(f'./{args.out_file}')