import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            cmd=f"./llama.cpp/build/bin/llama-quantize {str(gguf)} {args.out_folder}/{str(gguf.stem)[:-3]+method+".gguf"} {method}"
            print(cmd)
            # Use shell=True to handle multi-line commands and pipes
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                executable='/bin/bash'
            )
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"{line.rstrip()}")
                    logger.debug(f"{line.rstrip()}")
            
            return_code = process.wait()            
            if return_code != 0:
                logger.error(f"Stage quantize failed with exit code {return_code}")
            else:
                logger.info(f"Stage quantize completed successfully")
    
    elif args.quantize_method=="gptq" or args.quantize_method=="awq":
        from datasets import load_dataset
        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(1024))["text"]
        from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
        from gptqmodel.quantization import METHOD
        from transformers import AutoTokenizer
        
        model_path=args.input_folder
        model_name=next(Path(model_path).rglob("*.gguf")).stem
        tokenizer=AutoTokenizer.from_pretrained(model_path)
        if args.quantize_method=="gptq":
            quant_config = QuantizeConfig(
                quant_method=METHOD.GPTQ,
                bits=int(args.quantize_bpw), group_size=64,
                # mse=0.05, failsafe=None, # bad?
            )
        elif args.quantize_method=="awq":
            # it consumes 1GB/sec disk
            quant_config=QuantizeConfig(
                quant_method=METHOD.AWQ,
                bits=int(args.quantize_bpw),
                offload_to_disk=False,pack_impl="gpu",
            )
        model=GPTQModel.load(model_path,quantize_config=quant_config, device="cuda:0") # quantize with gpu
        model.quantize(calibration=calibration_dataset, backend=BACKEND.TRITON)
        model.save(f"{args.out_folder}")