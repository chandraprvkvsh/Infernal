#!/usr/bin/env python3
"""
Infernal - A lightweight local LLM inference and benchmarking tool with finetuning capabilities
"""

import os
import sys
import json
import click
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.panel import Panel
from tqdm import tqdm
import huggingface_hub
import re
import tempfile
import shutil
import subprocess
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
import platform
import urllib.request
import zipfile
import psutil
import time
import threading

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed. Run: pip install llama-cpp-python")
    sys.exit(1)

console = Console()

class InfernalLLM:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            homebrew_models = Path.home() / ".infernal-models"
            if homebrew_models.exists():
                models_dir = str(homebrew_models)
            else:
                models_dir = "models"
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / "config.json"
        self.load_config()

    def load_config(self):
        """Load or create configuration file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "models": {},
                "default_model": None,
                "settings": {
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            self.save_config()

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the local path for a model"""
        if model_name in self.config["models"]:
            model_path = self.models_dir / self.config["models"][model_name]["filename"]
            if model_path.exists():
                return model_path
        return None

    def download_model(self, model_url: str) -> Path:
        """Download a model from URL using Hugging Face hub"""
        try:
            if "huggingface.co" in model_url:
                parts = model_url.split("/")
                repo_id = f"{parts[3]}/{parts[4]}"
                filename_in_repo = parts[-1]
                model_name = repo_id.split("/")[-1]
                
                console.print(f"Detected Hugging Face repo: {repo_id}, file: {filename_in_repo}")
                console.print(f"Model name: {model_name}")
                
                filename = f"{model_name}.gguf"
                model_path = self.models_dir / filename

                if model_path.exists():
                    console.print(f"Model {model_name} already exists at {model_path}")
                    return model_path

                with console.status(f"Downloading {model_name} from Hugging Face..."):
                    downloaded_path = huggingface_hub.hf_hub_download(
                        repo_id=repo_id,
                        filename=filename_in_repo,
                        local_dir=self.models_dir,
                        local_dir_use_symlinks=False
                    )

                if Path(downloaded_path).exists():
                    Path(downloaded_path).rename(model_path)

                self.config["models"][model_name] = {
                    "filename": filename,
                    "repo_id": repo_id,
                    "original_filename": filename_in_repo,
                    "size": model_path.stat().st_size
                }
                self.save_config()
                console.print(f"Model {model_name} downloaded successfully!")
                return model_path
            else:
                console.print("Please provide a Hugging Face URL for automatic model name extraction")
                raise ValueError("Only Hugging Face URLs are supported")

        except Exception as e:
            console.print(f"Error downloading model: {e}")
            raise

    def download_from_huggingface(self, repo_id: str, filename: str) -> Path:
        """Download a model from Hugging Face"""
        model_name = repo_id.split("/")[-1]
        local_filename = f"{model_name}.gguf"
        model_path = self.models_dir / local_filename

        if model_path.exists():
            console.print(f"Model {model_name} already exists at {model_path}")
            return model_path

        console.print(f"Downloading {model_name} from Hugging Face ({repo_id})...")

        try:
            with console.status(f"Downloading {model_name} from Hugging Face..."):
                huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.models_dir,
                    local_dir_use_symlinks=False
                )

            downloaded_path = self.models_dir / filename
            if downloaded_path.exists():
                downloaded_path.rename(model_path)

            self.config["models"][model_name] = {
                "filename": local_filename,
                "repo_id": repo_id,
                "original_filename": filename,
                "size": model_path.stat().st_size
            }
            self.save_config()
            console.print(f"Model {model_name} downloaded successfully!")
            return model_path

        except Exception as e:
            console.print(f"Error downloading model: {e}")
            raise

    def list_models(self):
        """List all available models"""
        if not self.config["models"]:
            console.print("No models installed. Use 'infernal pull' to download a model.")
            return

        console.print("\nInstalled Models:")
        for name, info in self.config["models"].items():
            size_mb = info.get("size", 0) / (1024 * 1024)
            status = "✓" if (self.models_dir / info["filename"]).exists() else "✗"
            console.print(f" {status} {name} ({size_mb:.1f} MB)")

    def run_model(self, model_name: str, prompt: str = None, interactive: bool = False):
        """Run a model for inference"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            console.print(f"Model '{model_name}' not found. Use 'infernal pull' to download it.")
            return

        try:
            console.print(f"Loading model {model_name}...")
            llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=os.cpu_count()
            )

            if interactive:
                self.interactive_chat(llm, model_name)
            else:
                if not prompt:
                    prompt = Prompt.ask("Enter your prompt")

                response = llm(
                    prompt,
                    max_tokens=self.config["settings"]["max_tokens"],
                    temperature=self.config["settings"]["temperature"],
                    top_p=self.config["settings"]["top_p"],
                    stop=["User:", "\n\n"]
                )

                console.print(Panel(response["choices"][0]["text"], title="Response"))

        except Exception as e:
            console.print(f"Error running model: {e}")

    def interactive_chat(self, llm, model_name: str):
        """Interactive chat mode"""
        console.print(f"\nChat with {model_name} (type 'quit' to exit)")
        console.print("=" * 50)
        
        while True:
            try:
                user_input = Prompt.ask("\nYou")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not user_input.strip():
                    continue

                console.print("\nAssistant")
                with console.status("Thinking..."):
                    response = llm(
                        user_input,
                        max_tokens=self.config["settings"]["max_tokens"],
                        temperature=self.config["settings"]["temperature"],
                        top_p=self.config["settings"]["top_p"],
                        stop=["User:", "\n\n"]
                    )
                
                console.print(response["choices"][0]["text"])
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"Error: {e}")
        
        console.print("\nGoodbye!")

    def benchmark_model(self, model_name: str, prompts: list, repeat: int = 1):
        """Benchmark model speed and memory usage"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            console.print(f"Model '{model_name}' not found. Use 'infernal pull' to download it.")
            return

        try:
            console.print(f"Loading model {model_name} for benchmarking...")
            llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=os.cpu_count()
            )

            process = psutil.Process(os.getpid())
            ttfts = []
            throughput_rates = []
            total_token_rates = []
            peak_memories = []
            total_times = []
            all_total_tokens = []
            total_runs = repeat * len(prompts)

            for i in range(repeat):
                for prompt_idx, prompt in enumerate(prompts):
                    if repeat > 1:
                        run_label = f"Iteration {i + 1}/{repeat} for prompt: {prompt[:50]}..."
                    elif len(prompts) > 1:
                        run_label = f"Benchmarking prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}..."
                    else:
                        run_label = f"Benchmarking prompt: {prompt[:50]}..."
                    
                    console.print(run_label)

                    peak_memory = 0
                    stop_event = threading.Event()

                    def monitor_memory():
                        nonlocal peak_memory
                        while not stop_event.is_set():
                            mem = process.memory_info().rss / (1024 * 1024)
                            if mem > peak_memory:
                                peak_memory = mem
                            time.sleep(0.1)

                    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
                    monitor_thread.start()

                    start_time = time.perf_counter()
                    
                    response = llm(
                        prompt,
                        max_tokens=self.config["settings"]["max_tokens"],
                        temperature=self.config["settings"]["temperature"],
                        top_p=self.config["settings"]["top_p"],
                        stop=["User:", "\n\n"],
                        stream=True
                    )

                    first_token_time = None
                    full_response = ""
                    
                    for chunk in response:
                        content = chunk['choices'][0]['text']
                        full_response += content
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start_time

                    end_time = time.perf_counter()
                    total_time = end_time - start_time
                    ttft = first_token_time or total_time

                    prompt_tokens = len(llm.tokenize(prompt.encode('utf-8'), False, False)) if prompt else 0
                    completion_tokens = len(llm.tokenize(full_response.encode('utf-8'), False, False)) if full_response else 0
                    
                    generation_time = total_time - ttft if total_time > ttft else total_time
                    
                    if completion_tokens > 0 and generation_time > 0:
                        throughput = completion_tokens / generation_time
                    else:
                        throughput = 0.0

                    total_token_rate = (prompt_tokens + completion_tokens) / total_time if total_time > 0 else 0

                    stop_event.set()
                    monitor_thread.join()

                    ttfts.append(ttft)
                    throughput_rates.append(throughput)
                    total_token_rates.append(total_token_rate)
                    peak_memories.append(peak_memory)
                    total_times.append(total_time)
                    all_total_tokens.append(completion_tokens)

                    console.print(f" Time to first token: {ttft:.3f} seconds")
                    console.print(f" Generation time (After-TTFT): {generation_time:.3f} seconds")
                    console.print(f" Throughput (generated tok/sec): {throughput:.2f}")
                    console.print(f" Total tokens/sec: {total_token_rate:.2f}")
                    console.print(f" Peak memory usage: {peak_memory:.2f} MB")

            if total_runs > 1:
                sum_total_time = sum(total_times)
                sum_total_tokens = sum(all_total_tokens)
                avg_ttft = sum(ttfts) / total_runs
                avg_throughput = sum(throughput_rates) / total_runs
                avg_total_tokens = sum(total_token_rates) / total_runs
                avg_peak_memory = sum(peak_memories) / total_runs

                console.print("\nAverages across all runs:")
                console.print(f" Total runs: {total_runs}")
                console.print(f" Total tokens generated: {sum_total_tokens}")
                console.print(f" Total time required: {sum_total_time:.2f} seconds")
                console.print(f" Avg Time to first token: {avg_ttft:.3f} seconds")
                console.print(f" Avg throughput (generated tok/sec): {avg_throughput:.2f}")
                console.print(f" Avg total tokens/sec: {avg_total_tokens:.2f}")
                console.print(f" Avg Peak memory usage: {avg_peak_memory:.2f} MB")

        except Exception as e:
            console.print(f"Error during benchmarking: {e}")

    def remove_model(self, model_name: str):
        """Remove a model"""
        if model_name not in self.config["models"]:
            console.print(f"Model '{model_name}' not found")
            return

        model_path = self.get_model_path(model_name)
        if model_path and model_path.exists():
            model_path.unlink()
            console.print(f"Removed model file: {model_path}")

        del self.config["models"][model_name]
        self.save_config()
        console.print(f"Removed model '{model_name}' from configuration")


def parse_modelfile(modelfile_path):
    """Parse a Modelfile and return a dict of its instructions"""
    config = {
        'FROM': None,
        'PARAMETER': {},
        'SYSTEM': None,
        'MESSAGES': [],
        'HF_TOKEN': None,
        'TEMPLATE': None
    }

    with open(modelfile_path, 'r') as f:
        template_lines = []
        in_template = False
        for line in f:
            line = line.rstrip('\n')
            if not line or line.strip().startswith('#'):
                continue
            if line.startswith('FROM '):
                config['FROM'] = line[len('FROM '):].strip()
            elif line.startswith('PARAMETER '):
                param = line[len('PARAMETER '):].strip()
                key, value = param.split(' ', 1)
                config['PARAMETER'][key] = value
            elif line.startswith('SYSTEM '):
                config['SYSTEM'] = line[len('SYSTEM '):].strip()
            elif line.startswith('MESSAGE '):
                m = re.match(r'MESSAGE (\w+) (.+)', line)
                if m:
                    role, content = m.groups()
                    config['MESSAGES'].append({'role': role, 'content': content})
            elif line.startswith('HF_TOKEN '):
                config['HF_TOKEN'] = line[len('HF_TOKEN '):].strip()
            elif line.startswith('TEMPLATE '):
                in_template = True
                template_lines = [line[len('TEMPLATE '):].strip()]
            elif in_template:
                if line.strip() == '"""' or line.strip() == "'''":
                    in_template = False
                    config['TEMPLATE'] = '\n'.join(template_lines)
                else:
                    template_lines.append(line)

        if in_template:
            config['TEMPLATE'] = '\n'.join(template_lines)

    return config


@click.group()
@click.option('--models-dir', default='models', help='Directory to store models')
@click.version_option(version='1.0.0', prog_name='infernal')
@click.pass_context
def cli(ctx, models_dir):
    """Infernal - Lightweight local LLM inference and benchmarking"""
    ctx.ensure_object(dict)
    ctx.obj['infernal'] = InfernalLLM(models_dir)


@cli.command()
@click.option('--url', help='Hugging Face URL to download the model')
@click.option('--repo-id', help='Hugging Face repository ID')
@click.option('--filename', help='Filename in the repository')
@click.pass_context
def pull(ctx, url, repo_id, filename):
    """Download a model (model name is automatically extracted)"""
    infernal = ctx.obj['infernal']
    
    if url:
        infernal.download_model(url)
    elif repo_id and filename:
        infernal.download_from_huggingface(repo_id, filename)
    else:
        console.print("Please provide either --url or both --repo-id and --filename")
        console.print("\nExample:")
        console.print(" infernal pull --url https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf")
        console.print(" infernal pull --repo-id TheBloke/Llama-2-7B-Chat-GGUF --filename llama-2-7b-chat.Q4_K_M.gguf")


@cli.command()
@click.pass_context
def list(ctx):
    """List installed models"""
    infernal = ctx.obj['infernal']
    infernal.list_models()


@cli.command()
@click.argument('model_name')
@click.option('--prompt', '-p', help='Prompt to send to the model')
@click.option('--interactive', '-i', is_flag=True, help='Start interactive chat mode')
@click.pass_context
def run(ctx, model_name, prompt, interactive):
    """Run a model"""
    infernal = ctx.obj['infernal']
    infernal.run_model(model_name, prompt, interactive)


@cli.command()
@click.argument('model_name')
@click.pass_context
def remove(ctx, model_name):
    """Remove a model"""
    infernal = ctx.obj['infernal']
    infernal.remove_model(model_name)


@cli.command()
@click.option('--modelfile', required=True, type=click.Path(exists=True), help='Path to the Modelfile')
@click.option('--output', required=True, type=click.Path(), help='Path to save the finetuned model (GGUF)')
@click.option('--name', required=False, type=str, help='Name to register the finetuned model')
@click.option('--epochs', required=False, type=int, default=3, help='Number of training epochs')
@click.option('--batch-size', required=False, type=int, default=2, help='Batch size')
@click.option('--learning-rate', required=False, type=float, default=2e-5, help='Learning rate')
@click.pass_context
def finetune(ctx, modelfile, output, name, epochs, batch_size, learning_rate):
    """Finetune a model using a Modelfile"""
    console.print(f"Parsing Modelfile: {modelfile}")
    config = parse_modelfile(modelfile)

    def get_param(key, default, typ):
        val = config['PARAMETER'].get(key, default)
        if typ == bool:
            return str(val).lower() in ['true', '1', 'yes']
        try:
            return typ(val)
        except Exception:
            return default

    models_dir = ctx.obj['infernal'].models_dir
    model_name = name if name else Path(output).stem
    repo_id = config['FROM']
    messages = config['MESSAGES']
    hf_token = config.get('HF_TOKEN')

    lora = get_param('lora', False, bool)
    load_in_4bit = get_param('load_in_4bit', False, bool)
    load_in_8bit = get_param('load_in_8bit', False, bool)
    lora_r = get_param('lora_r', 8, int)
    lora_alpha = get_param('lora_alpha', 32, int)
    lora_dropout = get_param('lora_dropout', 0.05, float)
    lora_target_modules = config['PARAMETER'].get('lora_target_modules', 'q_proj,v_proj').split(',')

    console.print(f"Downloading base model and tokenizer from Hugging Face: {repo_id}")

    device_param = config['PARAMETER'].get('device', None)
    if device_param:
        device = device_param.lower()
        if device not in ['cuda', 'cpu']:
            console.print(f"Unknown device '{device}', defaulting to auto-detect.")
            device = None
    else:
        device = None

    if not device:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            device = 'cuda'
            console.print(f"GPU detected! Using CUDA with {n_gpus} GPU(s) for finetuning.")
        else:
            device = 'cpu'
            console.print("No GPU detected. Training will run on CPU (much slower).")
    else:
        if device == 'cuda' and not torch.cuda.is_available():
            console.print("Requested CUDA but no GPU found. Falling back to CPU.")
            device = 'cpu'

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            console.print("No pad_token found in tokenizer. Setting pad_token = eos_token.")

        model_kwargs = {}
        if lora:
            if load_in_4bit:
                model_kwargs['load_in_4bit'] = True
                model_kwargs['device_map'] = 'auto'
            elif load_in_8bit:
                model_kwargs['load_in_8bit'] = True
                model_kwargs['device_map'] = 'auto'

        model = AutoModelForCausalLM.from_pretrained(repo_id, token=hf_token, **model_kwargs)

        if lora:
            console.print(f"LoRA/PEFT enabled. Wrapping model with LoRA adapters (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}) targeting modules: {lora_target_modules}")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias='none',
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)

        model.to(device)

    except Exception as e:
        console.print(f"Error downloading model/tokenizer or applying LoRA: {e}")
        return

    console.print("Preparing dataset from Modelfile messages...")
    data = []
    for i in range(0, len(messages), 2):
        if i+1 < len(messages) and messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
            data.append({
                'instruction': messages[i]['content'],
                'output': messages[i+1]['content'],
                'user_message': messages[i],
                'assistant_message': messages[i+1]
            })

    if not data:
        console.print("No valid user/assistant message pairs found in Modelfile!")
        return

    formatted_data = []
    if hasattr(tokenizer, 'apply_chat_template'):
        console.print("Using tokenizer.apply_chat_template for prompt formatting...")
        for ex in data:
            chat_messages = [
                {"role": "user", "content": ex['instruction']},
                {"role": "assistant", "content": ex['output']}
            ]
            formatted_data.append({'text': tokenizer.apply_chat_template(chat_messages, tokenize=False)})
    else:
        template = config.get('TEMPLATE')
        system_prompt = config.get('SYSTEM')
        if not template:
            template = """{{ .System }}\nUser: {{ .Prompt }}\nAssistant: {{ .Response }}"""

        def render_template(system, prompt, response, template):
            result = template
            if system is not None:
                result = result.replace('{{ .System }}', system)
            else:
                result = result.replace('{{ .System }}\n', '').replace('{{ .System }}', '')
            result = result.replace('{{ .Prompt }}', prompt)
            result = result.replace('{{ .Response }}', response)
            return result

        for ex in data:
            formatted_data.append({'text': render_template(system_prompt, ex['instruction'], ex['output'], template)})

    dataset = Dataset.from_list(formatted_data)

    try:
        max_length = int(config['PARAMETER'].get('max_length', 2048))
    except Exception:
        max_length = 2048

    def preprocess(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_length)

    tokenized_dataset = dataset.map(preprocess, batched=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = f"{tmpdir}/finetuned_model"
        console.print("Starting Hugging Face Trainer finetuning...")

        max_length = get_param('max_length', 2048, int)
        learning_rate = get_param('learning_rate', 2e-5, float)
        epochs = get_param('epochs', epochs, int)
        batch_size = get_param('batch_size', batch_size, int)
        weight_decay = get_param('weight_decay', 0.0, float)
        warmup_steps = get_param('warmup_steps', 0, int)
        gradient_accumulation_steps = get_param('gradient_accumulation_steps', 1, int)
        fp16 = get_param('fp16', False, bool)
        save_steps = get_param('save_steps', 500, int)
        logging_steps = get_param('logging_steps', 5, int)
        lr_scheduler_type = config['PARAMETER'].get('lr_scheduler_type', 'linear')
        eval_steps = get_param('eval_steps', None, int)
        save_total_limit = get_param('save_total_limit', None, int)
        seed = get_param('seed', None, int)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            save_strategy='steps',
            save_steps=save_steps,
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            report_to=[],
            seed=seed if seed is not None else 42,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        try:
            trainer.train()
            trainer.save_model(output_dir)

            if lora:
                console.print("Saving LoRA adapter weights in models/ directory...")
                adapter_dir = models_dir / f"{model_name}-lora-adapter"
                model.save_pretrained(str(adapter_dir))

                console.print("Merging LoRA adapters into base model before GGUF conversion...")
                model = model.merge_and_unload()
                merged_dir = models_dir / f"{model_name}-merged"
                model.save_pretrained(str(merged_dir))
                output_dir = str(merged_dir)

            console.print(f"Finetuning complete! Your merged model is saved at: {output_dir}")
            console.print("To use your model with llama.cpp, convert it to GGUF using convert_hf_to_gguf.py. Example:")
            console.print(f"python3 convert_hf_to_gguf.py --in {output_dir} --out <model_name>.gguf")
            console.print("Then upload the GGUF file to your Hugging Face repo for easy download and use with llama.cpp!")

        except Exception as e:
            console.print(f"Error during training: {e}")
            return


@cli.command()
@click.argument('model', required=True)
@click.option('--prompt', default=None, help='Single prompt to benchmark')
@click.option('--promptfile', default=None, type=click.Path(exists=True), help='File with multiple prompts')
@click.option('--repeat', default=1, type=int, help='Number of times to repeat the benchmark')
@click.pass_context
def benchmark(ctx, model, prompt, promptfile, repeat):
    """Benchmark model performance with prompts"""
    infernal = ctx.obj['infernal']
    prompts = []

    if prompt and promptfile:
        console.print("Provide either --prompt or --promptfile, not both")
        return

    if prompt:
        if not prompt.strip():
            console.print("--prompt provided but empty")
            return
        prompts = [prompt]
        effective_repeat = repeat
    elif promptfile:
        if not Path(promptfile).exists():
            console.print("--promptfile provided but file does not exist")
            return
        config = parse_modelfile(promptfile)
        for msg in config['MESSAGES']:
            if msg['role'] == 'user':
                prompts.append(msg['content'])
        effective_repeat = 1
        if repeat > 1:
            console.print(f"Warning: --repeat={repeat} ignored for --promptfile (repetitions not supported for prompt files). Using repeat=1.")
    else:
        console.print("Provide either --prompt or --promptfile")
        return

    if not prompts:
        console.print("No prompts found")
        return

    infernal.benchmark_model(model, prompts, repeat=effective_repeat)


if __name__ == '__main__':
    cli()