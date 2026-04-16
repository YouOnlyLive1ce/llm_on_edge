import argparse
import subprocess
from pathlib import Path
from gptqmodel import GPTQModel, QuantizeConfig, get_best_device, BACKEND
from gptqmodel.quantization import METHOD
from transformers import AutoTokenizer
from datasets import load_dataset

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Execute quantization pipelines")

    parser.add_argument("input_folder")
    parser.add_argument("out_folder")
    parser.add_argument("quantize_method")
    parser.add_argument("quantize_bpw")
    args = parser.parse_args()
    
    # search for gguf in input_folder
    ggufs=list(Path(args.input_folder).rglob("*.gguf"))
    
    if args.quantize_method=="llama.cpp":
        print(f"found ggufs to quantize {ggufs}")
        
        method="IQ4_XS"
        if args.quantize_bpw=="4":
            method="IQ4_XS"
        elif args.quantize_bpw=="8":
            method="Q8_0"
        
        for gguf in ggufs:
            cmd=["./llama.cpp/build/bin/llama-quantize", str(gguf), f"{args.out_folder}/{str(gguf.stem)[:-3]+method+".gguf"}", method]
            print(cmd)
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                executable='/bin/bash'
            )
    elif args.quantize_method=="gptq" or args.quantize_method=="awq":
        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(1024))["text"]
        
        model_path=str(args.input_folder)
        out_folder=str(args.out_folder)
        quant_method=METHOD.GPTQ if args.quantize_method=="gptq" else METHOD.AWQ
        tokenizer=AutoTokenizer.from_pretrained(model_path)
        quant_config = QuantizeConfig(
            quant_method=quant_method,
            bits=int(args.quantize_bpw), group_size=32,
            mse=0.05, failsafe=None,
        )
        model=GPTQModel.load(model_path,quantize_config=quant_config, device="cuda:0") # quantize with gpu
        model.quantize(calibration=calibration_dataset, backend=BACKEND.TORCH)
        model.save(out_folder)    