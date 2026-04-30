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
        self.amount_samples=200
        self.dataset=load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        self.results={
                        'answers':[],
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
        self.results={'accuracy':[],'ppl':[],'flips':[],'prompt_per_second':[],'predicted_per_second':[],'answers':[]}
        
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
            self.results['answers'].append(answer)
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
        import pandas as pd
        import numpy as np
        import scipy.stats
        from pathlib import Path
        
        json_files = [str(path) for path in Path(path_to_jsons).rglob("*.json")]
        logger.info(f"{len(json_files)} json_files: {json_files}")
        dfs = []
        for file in json_files:
            df = pd.read_json(file, orient='index')
            df = df.transpose()
            df=df.drop(columns=["answers"])
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
            # print(df['accuracy'].notna().sum(),df['ppl'].notna().sum())

        # Find index of the base model
        base_model_idx = None
        for i, fname in enumerate(json_files):
            if fname == base_model_file_name:
                base_model_idx = i
                break
        if base_model_idx is None:
            raise ValueError(f"Base model file '{base_model_file_name}' not found in json_files")

        result_df=[]
        for i in range(len(json_files)):
            df = dfs[i]

            def mean_std(series):
                if series is None or len(series) == 0:
                    return "NaN"
                return f"{series.mean():.2f}+-{series.std():.2f}"

            ppl_str = mean_std(df['ppl'])
            prefill_str = mean_std(df['prompt_per_second'])
            decode_str = mean_std(df['predicted_per_second'])
            
            accuracy_mean = df['accuracy'].mean()
            base_acc = dfs[base_model_idx]['accuracy']
            curr_acc = df['accuracy']

            if i == base_model_idx:
                p_value = np.nan
                ci_lower, ci_upper = np.nan, np.nan
                a=(base_acc==1.0).sum()
                b=0.0
                c=0.0
                d=(base_acc==0.0).sum()
            else:
                a = ((base_acc == 1.0) & (curr_acc == 1.0)).sum()
                b = ((base_acc == 1.0) & (curr_acc == 0.0)).sum()
                c = ((base_acc == 0.0) & (curr_acc == 1.0)).sum()
                d = ((base_acc == 0.0) & (curr_acc == 0.0)).sum()
                if b + c == 0:
                    p_value = np.nan
                    ci_lower, ci_upper = np.nan, np.nan
                else:
                    # McNemar's test
                    chi2_stat = (b - c) ** 2 / (b + c)
                    p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, 1)
                    
                    # 95% confidence interval for difference in proportions
                    n = base_acc.size
                    diff = (b - c) / n
                    se = np.sqrt((b + c)) / n
                    ci_lower, ci_upper = diff - 1.96 * se, diff + 1.96 * se

            result_df.append({
                "method": Path(json_files[i]).name,
                "accuracy": f"{accuracy_mean:.2f}",
                "ppl": ppl_str,
                "prefill": prefill_str,
                "decode": decode_str,
                # "memory": memory_str,
                "correct->correct": f"{a/curr_acc.size:.2f}",
                "correct->incorrect": f"{b/curr_acc.size:.2f}",
                "incorrect->correct": f"{c/curr_acc.size:.2f}",
                "incorrect->incorrect":f"{d/curr_acc.size:.2f}",
                # "fail": f"{fail_count/df['accuracy'].size:.2f}",
                "mcNemar p value for accuracy": f"{p_value:.2f},n={df['accuracy'].notna().sum()}",
                "CI (95%)" : f"{ci_lower:.2f},{ci_upper:.2f}"
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
        for col in result_df.columns:
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
    parser.add_argument("--models-path", default="./out/olmo2", help="where ggufs and gptqmodels are")
    parser.add_argument("--results-path", default="./out/olmo2/benchmark_results", help="where results are")
    parser.add_argument("--server-url",default="http://localhost:8080")
    args = parser.parse_args()
    
    folder_path = Path(args.models_path)
    # run all ggufs (original_q16, pruned_q16, pruned_qX) and /quantized/model_{gptq|awq}X.safetensors
    
    models = [str(path) for path in folder_path.rglob("*.gguf")] # [str(path.parent) for path in folder_path.rglob("quantize_config.json")] + [str(path) for path in folder_path.rglob("*.gguf")]
    logger.info(f"models found: {models}")
    # models also can be hf link for remote server
    
    client=BenchLLMClient(args.server_url)
    for model in models:
        client.start_server(model)
        time.sleep(400) # waiting is performed ob server side
        client.bench_accuracy()
        # client.bench_perplexity()
        client.save_to_file(f'./{args.results_path}/{client.model_id}')
        client.stop_server()
        time.sleep(120)