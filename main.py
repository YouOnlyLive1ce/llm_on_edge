import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineExecutor:
    def __init__(self, config_path: str, model_name: str = None):
        self.config_path = config_path
        self.model_name = model_name
        self.config = None
        self.executed_stages = set()
        
    def load_config(self):
        """Load and parse the YAML configuration file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Update model name if provided
        if self.model_name:
            self.config['params']['huggingface_link'] = self.model_name
        
        # Resolve all paths
        self._resolve_paths()
    
    def _resolve_paths(self):
        """Replace placeholders in paths with actual values."""
        params = self.config['params']
        
        for stage_name, stage_config in self.config['stages'].items():
            # Resolve command
            if 'cmd' in stage_config:
                stage_config['cmd'] = self._replace_placeholders(stage_config['cmd'], params)
            
            # Resolve dependencies
            if 'deps' in stage_config:
                stage_config['deps'] = [self._replace_placeholders(dep, params) 
                                       for dep in stage_config['deps']]
            
            # Resolve outputs
            if 'outs' in stage_config:
                stage_config['outs'] = [self._replace_placeholders(out, params) 
                                       for out in stage_config['outs']]
    
    def _replace_placeholders(self, text: str, params: Dict) -> str:
        """Replace ${variable} placeholders with actual values."""
        for key, value in params.items():
            placeholder = f"${{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, str(value))
        return text
    
    def check_dependencies(self, stage_name: str) -> bool:
        """Check if all dependencies for a stage are satisfied."""
        stage_config = self.config['stages'][stage_name]
        
        if 'deps' not in stage_config:
            return True
        
        for dep in stage_config['deps']:
            # Check if it's a file/directory that needs to exist
            if not os.path.exists(dep):
                logger.warning(f"Dependency not found: {dep}")
                return False
        return True
    
    def check_outputs_exist(self, stage_name: str) -> bool:
        """Check if stage outputs already exist (for skipping)."""
        stage_config = self.config['stages'][stage_name]
        
        if 'outs' not in stage_config:
            return False
        
        for output in stage_config['outs']:
            if not os.path.exists(output):
                return False
        return True
    
    def execute_stage(self, stage_name: str, force: bool = False):
        """Execute a single pipeline stage."""
        if stage_name in self.executed_stages:
            logger.info(f"Stage '{stage_name}' already executed, skipping.")
            return True
        
        stage_config = self.config['stages'][stage_name]
        
        # Skip if outputs already exist and not forced
        if not force and self.check_outputs_exist(stage_name):
            logger.info(f"Outputs for '{stage_name}' already exist, skipping.")
            self.executed_stages.add(stage_name)
            return True
        
        logger.info(f"Executing stage: {stage_name}")
        
        # Handle for_each loops (like for quantize_method)
        if 'for_each' in stage_config:
            param_name = stage_config['for_each']
            param_values = self.config['params'].get(param_name, [])
            
            for value in param_values:
                logger.info(f"  Processing {param_name}: {value}")
                cmd = stage_config['cmd'].replace("${item}", value)
                outs = [out.replace("${item}", value) for out in stage_config.get('outs', [])]
                
                if not self._run_command(cmd, stage_name):
                    return False
                
                # Verify outputs were created
                for out in outs:
                    if out and not os.path.exists(out):
                        logger.warning(f"Expected output not created: {out}")
        else:
            # Regular single command execution
            cmd = stage_config['cmd']
            if not self._run_command(cmd, stage_name):
                return False
        
        self.executed_stages.add(stage_name)
        return True
    
    def _run_command(self, cmd: str, stage_name: str) -> bool:
        """Execute a shell command."""
        try:
            logger.debug(f"Running command: {cmd}")
            
            # Create output directory if needed
            output_dirs = ["models", "out", "benchmark/benchmark_results"]
            for dir_path in output_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Execute command
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Stage '{stage_name}' failed with exit code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return False
            
            logger.info(f"Stage '{stage_name}' completed successfully")
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing stage '{stage_name}': {e}")
            return False
    
    def execute_pipeline(self, stages: List[str] = None, force: bool = False):
        """Execute the entire pipeline or specific stages."""
        self.load_config()
        
        # Determine which stages to execute
        if stages is None:
            stages = list(self.config['stages'].keys())
        
        logger.info(f"Starting pipeline execution for stages: {stages}")
        
        for stage_name in stages:
            if stage_name not in self.config['stages']:
                logger.error(f"Unknown stage: {stage_name}")
                continue
            
            success = self.execute_stage(stage_name, force)
            if not success:
                logger.error(f"Pipeline failed at stage: {stage_name}")
                sys.exit(1)
        
        logger.info("Pipeline execution completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Execute the model optimization pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", help="Override model name from config")
    parser.add_argument("--stages", help="Comma-separated list of stages to execute")
    parser.add_argument("--force", action="store_true", help="Force re-execution of stages")
    parser.add_argument("--list-stages", action="store_true", help="List available stages")
    
    args = parser.parse_args()
    
    # Initialize executor
    executor = PipelineExecutor(args.config, args.model)
    executor.load_config()
    
    if args.list_stages:
        print("Available stages:")
        for stage in executor.config['stages'].keys():
            print(f"  - {stage}")
        return
    
    # Determine stages to execute
    stages_to_execute = None
    if args.stages:
        stages_to_execute = [s.strip() for s in args.stages.split(",")]
    
    # Execute pipeline
    executor.execute_pipeline(stages_to_execute, args.force)

if __name__ == "__main__":
    main()