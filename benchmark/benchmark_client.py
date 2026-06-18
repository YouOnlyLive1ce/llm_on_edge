from datasets import load_dataset
import requests
import json
import time
import logging
import os
import scipy
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchLLMClient:
    def __init__(self,server_url,wiki_path="./wikitext-2-raw/wiki.test.raw"):
        self.model_id=None
        self.server_type=None
        self.wiki_path=wiki_path
        self.amount_samples=200
        # self.dataset=load_dataset("hotpotqa/hotpot_qa", "fullwiki")
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
        
    def generate_questions(self):
        golden_dataset=[]
        for i in range(self.amount_samples):
            logger.info(i)
            row = self.dataset['validation'][i]
            golden_dataset.append({'user_question': \
                                f"Question context: {" ".join(row['context']['title'])}. \
                                Answer my question: {row['question']}. Correct answer:",
                                'golden_answer':row['answer']}
                                )
        os.makedirs(os.path.dirname(f'./golden_dataset.json'), exist_ok=True)
        with open(f'./golden_dataset.json','w') as json_file:
            json.dump(golden_dataset,json_file,indent=2)
    
    def llm_as_judge():
        # call imported function
        ...
    
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
            # self.results['accuracy'].append(any(word in answer.split() for word in row['answer'].split()))
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
        os.makedirs(os.path.dirname(f'./{filename}.json'), exist_ok=True)
        with open(f'./{filename}.json','w') as json_file:
            json.dump(self.results,json_file,indent=2)
    
    def preprocess(self,text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        stop_words.discard("yes")
        stop_words.discard("no")
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and stopwords, then lemmatize
        processed = set()
        for token in tokens:
            if token not in string.punctuation and token not in stop_words:
                processed.add(lemmatizer.lemmatize(token))
        
        return processed
    
    def calc_scores(self, candidate_answers, golden_df, metric='precision'):
        """
        candidate_answers: List of predicted answer strings
        golden_df: DataFrame with 'golden_answer' column
        metric: 'precision', 'recall', or 'f1'
        """
        
        scores = []
        
        for i in range(len(candidate_answers)):
            cand_set = self.preprocess(candidate_answers[i])
            gold_set = self.preprocess(golden_df["golden_answer"][i])
            
            if not cand_set or not gold_set:
                scores.append(0.0)
                continue
            
            intersection = cand_set.intersection(gold_set)
            
            if metric == 'precision':
                # Proportion of candidate words that are correct
                score = len(intersection) / len(cand_set)
            elif metric == 'recall':
                # Proportion of golden words captured
                score = len(intersection) / len(gold_set)
            elif metric == 'f1':
                # Harmonic mean of precision and recall
                precision = len(intersection) / len(cand_set) if cand_set else 0
                recall = len(intersection) / len(gold_set) if gold_set else 0
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                raise ValueError("metric must be 'precision', 'recall', or 'f1'")
            
            scores.append(score)
        
        return scores
    
    def calc_categories(self, candidate_answers, golden_df):
        categories = []
        f1s=self.calc_scores(candidate_answers,golden_df,'f1')
        for i in range(len(candidate_answers)):
            cand_set = self.preprocess(candidate_answers[i])
            gold_set = self.preprocess(golden_df["golden_answer"][i])
            
            intersection = cand_set.intersection(gold_set)
            
            category="d_judge_fail"
            if len(candidate_answers[i])==0:
                category="d_empty"
            elif not cand_set or not gold_set:
                category="d_malformed"
            elif len(self.preprocess("question context answer").intersection(cand_set))>0:
                category="d_dataset_continuation"
            elif "?" in candidate_answers[i]:
                category="d_question_gen"
            elif len(intersection) / len(gold_set)==0:
                category="d_incorrect"
            elif f1s[i]>0.1:
                category="d_correct"
            elif f1s[i]<0.1:
                category="d_no_final"
            categories.append(category)
            
        return categories
    
    def compare_results(self, path_to_jsons, base_model_file_name, golden_path="./golden_dataset.json"):
        import pandas as pd
        from pathlib import Path
        
        json_files = [str(path) for path in Path(path_to_jsons).rglob("*.jsona")]
        # Find index of the base model
        base_model_idx = None
        for i, fname in enumerate(json_files):
            if fname == base_model_file_name:
                base_model_idx = i
                break
        if base_model_idx is None:
            raise ValueError(f"Base model file '{base_model_file_name}' not found in json_files")

        dfs = []
        golden_df=pd.read_json(golden_path)
        precisions=[]
        recalls=[]
        f1s=[]
        determenistic_categories=[]
        n=len(golden_df["golden_answer"])
        for file in json_files:
            df = pd.read_json(file, orient='index')
            df = df.transpose()
            if df['ppl'].dtype==object: #todo remove if
                corrected = df['ppl'].str.extract(r'\](\d+\.?\d*)')
                corrected = corrected[0]  # Extract from DataFrame to Series
                mask = corrected.notna()
                df.loc[mask, 'ppl'] = corrected[mask]
                df['ppl'] = pd.to_numeric(df['ppl'], errors='coerce')
            # df[~np.isfinite(df['ppl'])]=df[np.isfinite(df['ppl'])].mean() 
            df['ppl'] = pd.to_numeric(df['ppl'])
            candidate_answers=df['answers']
            precisions.append(self.calc_scores(candidate_answers,golden_df,'precision'))
            recalls.append(self.calc_scores(candidate_answers,golden_df,'recall'))
            f1s.append(self.calc_scores(candidate_answers,golden_df,'f1'))
            determenistic_categories.append(self.calc_categories(candidate_answers,golden_df))
            dfs.append(df)

        aggregated_metrics_df=[]
        aggregated_metrics_analysis_df=[]
        for i in range(len(json_files)):
            df = dfs[i]
            raw_metrics_df=pd.concat([pd.DataFrame(list(zip(precisions[i],recalls[i],f1s[i],determenistic_categories[i],
                                    [self.preprocess(strr) for strr in df['answers']],
                                    [self.preprocess(cand_strr).intersection(self.preprocess(gold_strr)) for cand_strr,gold_strr in \
                                        zip(df['answers'],golden_df['golden_answer'])]
                                    )),
                                    columns=["precision", "recall", "f1", "d_category", "normalized_answer", "extracted answer"]),
                                    df,
                                    golden_df
                                    ],
                                    axis=1)
            Path(f"../../out/raw_results").mkdir(parents=True,exist_ok=True)
            # extensive results on which figures in paper are build
            raw_metrics_df.to_json(f"../../out/raw_results/{Path(json_files[i]).name}",orient='records',indent=2)
            
            def mean_std(series):
                if series is None or len(series) == 0:
                    return "NaN"
                return f"{series.mean():.2f}+-{series.std():.2f}"

            ppl_str = mean_std(df['ppl'])
            prefill_str = mean_std(df['prompt_per_second'])
            decode_str = mean_std(df['predicted_per_second'])
            
            base_category = dfs[base_model_idx]['category']
            curr_category = df['category']
            
            base_transitions=pd.crosstab(base_category, curr_category, colnames=["category"])
            # mcnemar compared to base model
            base_a=base_transitions.get("CORRECT").get("CORRECT").item()
            base_b=base_transitions.get("CORRECT").get("INCORRECT").item()
            base_c=base_transitions.get("INCORRECT").get("CORRECT").item()
            base_d=base_transitions.get("INCORRECT").get("INCORRECT").item()
            
            # mcnemar comparison of current treatment (base->pruning->quant) vs no treatment (base->quant)
            treatment_p_val=None
            treatment_std_precision=None
            treatment_std_recall=None
            treatment_delta_precision=None
            treatment_delta_recall=None
            curr_no_prune_category=None
            pure_quantized_idx=None
            treatment_a=(df['category']=="CORRECT").sum()
            treatment_b=0
            treatment_c=0
            treatment_d=(df['category']=="INCORRECT").sum()
            if "quantized" in json_files[i] and ("sparsegpt" in json_files[i] or "wanda" in json_files[i]):
                # corresponding no treatment experiment
                pure_quantized=json_files[i].replace("unstructured_","").replace("sparsegpt_","").replace("wanda_","").replace("0.2_","").replace("0.5_","")
                for j in range(len(json_files)):
                    if pure_quantized==json_files[j]:
                        pure_quantized_idx=j
                        break
                
                curr_no_prune_category=dfs[pure_quantized_idx]['category']
                treatment_transitions=pd.crosstab(curr_no_prune_category,curr_category, colnames=["category"])
                # mcnemar compared treatment to no treatment
                print("comparing treatment model",json_files[i]," to no treatment (no pruning) model",json_files[pure_quantized_idx])
                treatment_a=treatment_transitions.get("CORRECT").get("CORRECT").item()
                treatment_b=treatment_transitions.get("CORRECT").get("INCORRECT").item()
                treatment_c=treatment_transitions.get("INCORRECT").get("CORRECT").item()
                treatment_d=treatment_transitions.get("INCORRECT").get("INCORRECT").item()
                treatment_p_val=1 - scipy.stats.chi2.cdf((treatment_b-treatment_c)**2 / (treatment_b+treatment_c), 1)
                
                # precision ci compared to no treatment model
                tp1=sum(precisions[pure_quantized_idx])/n
                tp2=sum(precisions[i])/n
                tn1=len(golden_df) # precision was calculated for every response
                tn2=len(golden_df) # precision was calculated for every response
                treatment_std_precision=(tp1*(1-tp1)/tn1 + tp2*(1-tp2)/tn2)**0.5
                
                # recall ci compared to treatment model
                tr1=sum(recalls[pure_quantized_idx])/n
                tr2=sum(recalls[i])/n
                tn1=len(golden_df) # recall was calculated for every response
                tn2=len(golden_df) # recall was calculated for every response
                treatment_std_recall=(tr1*(1-tr1)/tn1 + tr2*(1-tr2)/tn2)**0.5
                
                treatment_delta_precision=sum(precisions[i])/n-sum(precisions[pure_quantized_idx])/n
                treatment_delta_recall=sum(recalls[i])/n-sum(recalls[pure_quantized_idx])/n
            
            name=Path(json_files[i]).name
            aggregated_metrics_df.append({
                "name": name,
                "ppl": ppl_str,
                "prefill": prefill_str,
                "decode": decode_str,
                "precision":sum(precisions[i])/n,
                "recall":sum(recalls[i])/n,
                "f1":sum(f1s[i])/n,
                "s_correct":(df['category']=="CORRECT").sum()/n,
                "s_incorrect":(df['category']=="INCORRECT").sum()/n,
                "s_malformed":(df['category']=="MALFORMED").sum()/n,
                "s_empty": (df['category']=="EMPTY").sum()/n,
                "s_no_final": (df['category']=="NO_FINAL_ANSWER").sum()/n,
                "s_question_gen": (df['category']=="QUESTION_GENERATED").sum()/n,
                "s_judge_fail": (df['category']=="JUDGE_FAIL").sum()/n,
                "d_correct":determenistic_categories[i].count("d_correct")/n,
                "d_incorrect": determenistic_categories[i].count("d_incorrect")/n,
                "d_malformed":determenistic_categories[i].count("d_malformed")/n,
                "d_empty": determenistic_categories[i].count("d_empty")/n,
                "d_no_final": determenistic_categories[i].count("d_no_final")/n,
                "d_question_gen": determenistic_categories[i].count("d_question_gen")/n,
                "d_dataset_cont":determenistic_categories[i].count("d_dataset_continuation")/n,
                "d_judge_fail": determenistic_categories[i].count("d_judge_fail")/n,
            })
            
            # p1=(base_category=="CORRECT").sum()/len(golden_df["golden_answer"])
            # p2=(curr_category=="CORRECT").sum()/len(golden_df["golden_answer"])
            # n1=(base_category=="CORRECT").sum()+ (base_category=="INCORRECT").sum()
            # n2=(curr_category=="CORRECT").sum()+ (curr_category=="INCORRECT").sum()
            
            # precision ci compared to base model
            base_p1=sum(precisions[base_model_idx])/n
            base_p2=sum(precisions[i])/n
            base_n1=len(golden_df) # precision was calculated for every response
            base_n2=len(golden_df) # precision was calculated for every response
            base_std_precision=(base_p1*(1-base_p1)/base_n1 + base_p2*(1-base_p2)/base_n2)**0.5
            
            base_p_val=1 - scipy.stats.chi2.cdf((base_b-base_c)**2 / (base_b+base_c), 1) if base_b+base_c>0 else 1
            
            # recall ci compared to base model
            base_r1=sum(recalls[base_model_idx])/n
            base_r2=sum(recalls[i])/n
            base_n1=len(golden_df) # precision was calculated for every response
            base_n2=len(golden_df) # precision was calculated for every response
            base_std_recall=(base_r1*(1-base_r1)/base_n1 + base_r2*(1-base_r2)/base_n2)**0.5
            
            aggregated_metrics_analysis_df.append({
                "name":name,
                
                "base cor->cor": base_a, #base to optimized
                "base cor->incor":base_c,#base to optimized
                "base incor->cor":base_b,
                "base incor->incor":base_d,
                "base instability":base_c+base_b,
                "base delta s_accuracy": (curr_category=="CORRECT").sum()/n - (base_category=="CORRECT").sum()/n,
                "base delta d_accuracy":determenistic_categories[i].count("d_correct")/n - determenistic_categories[base_model_idx].count("d_correct")/n,
                "base delta precision":base_p2-base_p1,
                "base delta recall":base_r2-base_r1,
                "base ci precision": f"{base_p2 - 1.96 * base_std_precision:.2f}..{
                                   base_p2 + 1.96 * base_std_precision:.2f}",
                "base ci recall":f"{base_r2 - 1.96 * base_std_recall:.2f}..{
                                   base_r2 + 1.96 * base_std_recall:.2f}",
                "base mcnemar p": str(f"{base_p_val:.2f}") if base_p_val<0.1 else ">", # represents that base and optimized distributions differ
                
                "treatment cor->cor": treatment_a, #base to optimized
                "treatment cor->incor":treatment_c,#base to optimized
                "treatment incor->cor":treatment_b,
                "treatment incor->incor":treatment_d,
                "treatment instability":treatment_b+treatment_c,
                "treatment delta s_accuracy":(curr_category=="CORRECT").sum()/n - (curr_no_prune_category=="CORRECT").sum()/n if curr_no_prune_category is not None else "-",
                "treatment delta d_accuracy":determenistic_categories[i].count("d_correct")/n - determenistic_categories[pure_quantized_idx].count("d_correct")/n if pure_quantized_idx else "-",
                "treatment delta precision":treatment_delta_precision if treatment_delta_precision else "-",
                "treatment delta recall":treatment_delta_recall if treatment_delta_recall else "-",
                "treatment ci precision": f"{base_p2 - 1.96 * treatment_std_precision:.2f}..{
                                   base_p2 + 1.96 * treatment_std_precision:.2f}" if treatment_std_precision else "-",
                "treatment ci recall": f"{base_r2 - 1.96 * treatment_std_recall:.2f}..{
                                   base_r2 + 1.96 * treatment_std_recall:.2f}" if treatment_std_recall else "-",
                "treatment mcnemar p": str(f"{treatment_p_val:.2f}") if treatment_p_val else "-" # represents that pruned and pruned->quantized distributions differ
            })

        aggregated_metrics_df = pd.DataFrame(aggregated_metrics_df).sort_values(by="name",key=lambda x: x.str[::-1]).reset_index(drop=True)
        analysis_df=pd.DataFrame(aggregated_metrics_analysis_df).sort_values(by="name",key=lambda x: x.str[::-1]).reset_index(drop=True)
        return aggregated_metrics_df, analysis_df
    
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
    
    def plot_tradeoffs(self, x, *ys, group_labels = None, names=None, colors=None, titles=None, xlabel='base instability', figsize=(16, 4), ncols=4):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
        
        n_plots = len(ys)
        nrows = 1
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        n_groups = len(names)
        
        # Plot each subplot
        for idx, y in enumerate(ys):
            if idx < len(axes):  # Ensure we don't exceed available axes
                ax = axes[idx]
                
                # Plot each group
                for group in range(n_groups):
                    mask = [g == group for g in group_labels]
                    ax.scatter(np.array(x)[mask], np.array(y)[mask], 
                            color=colors[group % len(colors)], 
                            label=names[group % len(names)])
                
                # Set title and labels
                if idx < len(titles):
                    ax.set_title(titles[idx])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(titles[idx] if titles and idx < len(titles) else f'y{idx+1}')
        
        # Create custom legend handles
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=colors[i % len(colors)], 
                                label=names[i % len(names)], markersize=10) 
                        for i in range(n_groups)]
        
        # Add a single legend for the entire figure
        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 0.5))
        
        plt.tight_layout()
        plt.show()

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
    
    models = [str(path.parent) for path in folder_path.rglob("quantize_config.json")] + [str(path) for path in folder_path.rglob("*.gguf")]
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