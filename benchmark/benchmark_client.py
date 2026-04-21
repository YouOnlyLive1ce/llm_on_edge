from datasets import load_dataset
import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchLLMClient:
    def __init__(self,server_url,wiki_path="./wikitext-2-raw/wiki.test.raw"):
        self.model_id=None
        self.server_type=None
        self.wiki_path=wiki_path
        self.amount_samples=2
        self.dataset=load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        self.results={
                        'accuracy':[],
                        'ppl':[],
                        'flips':[],
                        'prompt_per_second':[],
                        'predicted_per_second':[]
                    }
        self.server_url=server_url
    
    def start_server(self,model_id):
        self.model_id=model_id
        self.server_type="llama.cpp" if "gguf" in self.model_id else "gptqmodel"
        headers = {"Content-Type": "application/json"}
        data={"model_id":model_id,"server_type":self.server_type,"status":"start"}
        requests.post(f'{self.server_url}/server_setup',headers=headers,json=data)
        logger.info(f"start server {model_id} with {self.server_type} framework")
        # wait for response setup?
    
    def stop_server(self):
        headers = {"Content-Type": "application/json"}
        data={"server_type":self.server_type,"status":"stop"}
        requests.post(f'{self.server_url}/server_setup',headers=headers,json=data)
        self.model_id=None
        self.server_type=None
        self.results={'accuracy':[],'ppl':[],'flips':[],'prompt_per_second':[],'predicted_per_second':[]}
        
    def bench_accuracy(self):
        headers = {"Content-Type": "application/json"}

        for i in range(self.amount_samples):
            logger.info(i)
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
            
            response = requests.get(f"{self.server_url}/bench_v1_chat_completions", 
                                    headers=headers, 
                                    json=data).json()
            
            # save stats
            logger.info(response)
            answer = response['chat']['content']
            self.results['accuracy'].append(any(word in answer.split() for word in row['answer'].split()))
            self.results['prompt_per_second'].append(response['prompt_per_second'])
            self.results['predicted_per_second'].append(response['predicted_per_second'])
    
    def bench_perplexity(self):
        # Setup data
        with open(self.wiki_path) as f:
            raw_texts = f.readlines()
        
        texts = []
        for line in raw_texts:
            if len(line) > 100:
                texts.append(line[:8000])
        
        data = {
            "texts": texts[:self.amount_samples],
            "amount": self.amount_samples
        }
        
        logger.info("start ppl")
        try:
            response = requests.post(  # Changed to POST
                f"{self.server_url}/bench_perplexity",
                json=data,
            )
            response.raise_for_status()  # Raise exception for bad status codes
            result = response.json()
            logger.info("end ppl")
            self.results['ppl']=result
        except requests.exceptions.Timeout:
            logger.info("Request timed out after 300 seconds")
        except requests.exceptions.RequestException as e:
            logger.info(f"Request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.info(f"Response: {e.response.text}")
    
    def save_to_file(self,filename):
        import os
        os.makedirs(os.path.dirname(f'./{filename}.json'), exist_ok=True)
        with open(f'./{filename}.json','w') as json_file:
            json.dump(self.results,json_file,indent=2)
    
    def compare_results(self, path_to_jsons, base_model_file_name):
        import os
        import pandas as pd
        import numpy as np
        import scipy.stats

        json_files = [pos_json for pos_json in os.listdir(path_to_jsons) if pos_json.endswith('.json')]
        logger.info(f"json_files: {json_files}")
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
            logger.info(df['accuracy'].notna().sum(),df['ppl'].notna().sum())

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
                logger.info("correct->incorrect",b)
                logger.info("incorrect->correct",c)
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
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Execute the model optimization pipeline")
    parser.add_argument("--models-path", default="./out/smollm2", help="where ggufs and gptqmodels are")
    parser.add_argument("--results-path", default="./out/benchmark_results2", help="where results are")
    parser.add_argument("--server-url",default="http://localhost:8080")
    args = parser.parse_args()
    
    folder_path = Path(args.models_path)
    # run all ggufs (original_q16, pruned_q16, pruned_qX) and /quantized/model_{gptq|awq}X.safetensors
    models = [str(path) for path in folder_path.rglob("*.gguf")] + [str(path.parent) for path in folder_path.rglob("*.safetensors")]
    logger.info(f"models found: {models}")
    # models also can be hf link for remote server
    
    client=BenchLLMClient(args.server_url)
    for model in models[2:]:
        client.start_server(model)
        time.sleep(300) # waiting is performed ob server side
        client.bench_accuracy()
        client.bench_perplexity()
        client.save_to_file(f'./{args.results_path}/{client.model_id}')
        client.stop_server()
        time.sleep(100)