# Infernal

**Infernal** is a powerful yet lightweight tool for running Large Language Models (LLMs) locally with blazing fast inference, comprehensive benchmarking, and model finetuning capabilities. Built on top of llama.cpp for maximum performance.

## Features

- **Fast Local Inference**: Lightning-fast model execution using llama.cpp
- **Simple Downloads**: Pull models directly from HuggingFace with zero configuration
- **Interactive Chat**: Engage in real-time conversations with your models
- **Performance Benchmarking**: Detailed performance metrics and analysis
- **Model Finetuning**: Train custom models using LoRA/PEFT with simple Modelfiles
- **Clean Management**: Simple model organization with automatic configuration
- **Zero Setup**: No complex configuration files - just download and run

## Quick Start

### Installation

**Method 1: Using pip (Recommended)**
```bash
git clone https://github.com/chandraprvkvsh/Infernal.git
cd infernal
pip install -e .
```

**Method 2: Direct installation**
```bash
git clone https://github.com/chandraprvkvsh/Infernal.git
cd infernal
pip install -r requirements.txt
```

After Method 1, you can use the `infernal` command globally. For Method 2, use `python infernal.py`.

### Basic Usage

**Download a model:**

Using pip installation:
```bash
infernal pull --repo-id TheBloke/Llama-2-7B-Chat-GGUF --filename llama-2-7b-chat.Q4_K_M.gguf
```

Or with direct URL:
```bash
infernal pull --url https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

Using direct method:
```bash
python infernal.py pull --repo-id TheBloke/Llama-2-7B-Chat-GGUF --filename llama-2-7b-chat.Q4_K_M.gguf
```

**List your models:**
```bash
infernal list
```

or
```bash
python infernal.py list
```

**Run inference:**

Single prompt:
```bash
infernal run llama-2-7b-chat.Q4_K_M.gguf --prompt "Explain quantum computing in simple terms"
```

Interactive chat:
```bash
infernal run llama-2-7b-chat.Q4_K_M.gguf --interactive
```

Using direct method:
```bash
python infernal.py run llama-2-7b-chat.Q4_K_M.gguf --prompt "Explain quantum computing"
```

**Benchmark performance:**

```bash
infernal benchmark llama-2-7b-chat.Q4_K_M.gguf --prompt "Write a short story about AI" --repeat 5
```

Using direct method:
```bash
python infernal.py benchmark llama-2-7b-chat.Q4_K_M.gguf --prompt "Write a story" --repeat 3
```

**Remove models:**

```bash
infernal remove llama-2-7b-chat.Q4_K_M.gguf
```

or
```bash
python infernal.py remove llama-2-7b-chat.Q4_K_M.gguf
```

## Model Finetuning

One of Infernal's most powerful features is the ability to finetune LLMs using a simple Modelfile approach, similar to Docker's approach but for AI models.

### Creating a Modelfile

A **Modelfile** is a simple text file that describes how to finetune a model. It contains the base model, training parameters, and example conversations.

Here's a complete example (`Modelfile`):

```
# Infernal Modelfile for Customer Support Bot
# Your HuggingFace token (required for private models)
HF_TOKEN your_huggingface_token_here

# Base model to finetune
FROM meta-llama/Llama-3.2-1B-Instruct

# Training configuration
PARAMETER device cuda
PARAMETER max_length 512
PARAMETER learning_rate 3e-5
PARAMETER epochs 3
PARAMETER batch_size 2
PARAMETER weight_decay 0.01
PARAMETER warmup_steps 100

# LoRA configuration for efficient training
PARAMETER lora true
PARAMETER lora_r 8
PARAMETER lora_alpha 32
PARAMETER lora_dropout 0.05
PARAMETER lora_target_modules q_proj,v_proj,k_proj,o_proj

# System prompt
SYSTEM You are a helpful customer support assistant for TechCorp. Be friendly, professional, and provide clear solutions.

# Training examples (as many as you need)
MESSAGE user How do I reset my password?
MESSAGE assistant To reset your password: 1) Go to the login page 2) Click "Forgot Password" 3) Enter your email 4) Check your email for reset instructions 5) Follow the link to create a new password. If you need further help, please contact our support team.

MESSAGE user My order hasn't arrived yet. What should I do?
MESSAGE assistant I understand your concern about your delayed order. Let me help you: 1) Check your email for tracking information 2) Use the tracking number on our website 3) If it's been more than 5 business days, please provide your order number and I'll investigate further. We apologize for any inconvenience.

MESSAGE user How do I return a defective product?
MESSAGE assistant For defective product returns: 1) Visit our Returns page 2) Select "Defective Item" as the reason 3) Upload photos of the defect 4) Print the prepaid return label 5) Package securely and ship back. You'll receive a replacement or refund within 5-7 business days after we receive the item.

MESSAGE user What's your refund policy?
MESSAGE assistant Our refund policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds typically process within 3-5 business days after we receive the returned item. Digital products and personalized items are non-refundable. Would you like help starting a return?
```

### Modelfile Parameters Explained

**Model Configuration:**
- `FROM`: Base model from HuggingFace (required)
- `HF_TOKEN`: Your HuggingFace access token
- `SYSTEM`: System prompt that defines the assistant's role

**Training Parameters:**
- `device`: Training device (`cuda` for GPU, `cpu` for CPU)
- `epochs`: Number of training cycles (default: 3)
- `batch_size`: Training batch size (default: 2)
- `learning_rate`: Learning rate (default: 2e-5)
- `max_length`: Maximum sequence length (default: 2048)
- `weight_decay`: Weight decay for regularization
- `warmup_steps`: Number of warmup steps

**LoRA/PEFT Parameters (for efficient training):**
- `lora`: Enable LoRA training (`true`/`false`)
- `lora_r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA alpha parameter (default: 32)
- `lora_dropout`: LoRA dropout rate (default: 0.05)
- `lora_target_modules`: Target modules for LoRA (e.g., `q_proj,v_proj`)

### Running Finetuning

Once you have your Modelfile ready:

**Basic finetuning**
```bash
infernal finetune --modelfile Modelfile --output my-custom-bot
```

**Override parameters from command line**
```bash
infernal finetune --modelfile Modelfile --output my-custom-bot --epochs 5 --batch-size 4
```

**Specify a custom name for the model**
```bash
infernal finetune --modelfile Modelfile --output my-custom-bot --name "customer-support-v1"
```

### After Finetuning

The finetuning process will:

1. Download the base model from HuggingFace
2. Prepare training data from your MESSAGE examples
3. Apply LoRA adapters for efficient training (if enabled)
4. Train the model using HuggingFace Trainer
5. Merge adapters back into the base model
6. Save the result in standard Hugging Face format

You can optionally convert the merged model to GGUF format using `llama-cpp`, and push the converted model to Hugging Face if desired.

Then use your custom model:

```bash
infernal pull --url https://huggingface.co/my-username/my-custom-bot/resolve/main/my-custom-bot.gguf
infernal run my-custom-bot --interactive
```

### Finetuning Tips

**For better results:**
- Use 10-50 example conversations in your Modelfile
- Keep examples focused on your specific use case
- Use consistent formatting in your MESSAGE blocks
- Test with small models first (1B-3B parameters)

**For efficient training:**
- Enable LoRA (`PARAMETER lora true`) to reduce memory usage
- Use smaller batch sizes if you run out of memory
- Start with fewer epochs (3-5) to avoid overfitting

**Hardware recommendations:**
- CPU training: Works but slow, use small models and batch_size=1
- GPU training: Much faster, can use larger models and batch sizes
- Memory: 8GB+ RAM minimum, 16GB+ recommended

## Benchmarking

Infernal provides comprehensive performance analysis with accurate timing measurements:

### Key Metrics
- Time to First Token (TTFT): Latency before first response token
- Throughput: Tokens generated per second during generation phase
- Total Token Rate: Combined input/output processing speed
- Memory Usage: Peak RAM consumption during inference
- Generation Time: Pure text generation time (excluding TTFT)

### Real-World Timing
Infernal uses actual measured timings without artificial thresholds, providing authentic performance data that reflects real-world usage patterns.

### Example Output
```
Run 1/3 - Prompt 1
Time to first token: 0.245 seconds
Generation time (After-TTFT): 2.156 seconds
Throughput (generated tok/sec): 23.45
Total tokens/sec: 19.87
Peak memory usage: 8547.23 MB

Averages across all runs:
Total runs: 3
Total tokens generated: 150
Total time required: 6.78 seconds
Avg Time to first token: 0.251 seconds
Avg throughput (generated tok/sec): 22.89
Avg total tokens/sec: 19.34
Avg Peak memory usage: 8521.45 MB
```

### Multiple Iterations
Run benchmarks multiple times for statistical accuracy:
```bash
infernal benchmark model.gguf --prompt "Test prompt" --repeat 10
```

### Benchmark with Modelfiles
You can also benchmark using prompts from a Modelfile:
```bash
infernal benchmark model.gguf --promptfile Modelfile
```

## Installation Methods Explained

### Method 1: Pip Installation (pyproject.toml)

This project uses modern Python packaging with `pyproject.toml` instead of the traditional `setup.py`. This provides:

- Cleaner dependency management: All project metadata in one file
- Modern build system: Uses `hatchling` as the build backend
- Automatic script creation: `infernal` command is automatically available system-wide
- Better development workflow: Use `pip install -e .` for editable installs

**Benefits:**
- Run `infernal` command from anywhere
- Cleaner project structure
- Modern Python packaging standards
- Easy uninstallation with `pip uninstall infernal`

### Method 2: Direct Requirements

Traditional approach using `requirements.txt` for those who prefer direct control:

- Manual dependency installation: `pip install -r requirements.txt`
- Direct script execution: `python infernal.py [command]`
- No system-wide installation: Runs only in project directory

**When to use each method:**
- Use Method 1 for permanent installation and system-wide access
- Use Method 2 for development, testing, or temporary usage

## Advanced Configuration

### Model Storage
By default, models are stored in a `models/` directory with a `config.json` file for management. Change the storage location with:
```bash
infernal --models-dir /path/to/models pull --url [URL]
```

### Performance Settings
Models are configured automatically, but you can modify settings in `models/config.json`:
```json
{
  "models": {},
  "default_model": null,
  "settings": {
    "max_tokens": 2048,
    "temperature": 0.2,
    "top_p": 0.7
  }
}
```

### Supported Model Formats
- GGUF files: Primary format for llama.cpp compatibility
- Quantized models: Q4_K_M, Q5_K_M, Q8_0, etc.
- HuggingFace integration: Direct downloads from HF repositories

## System Requirements

### Minimum Requirements
- Python 3.9+
- 4GB RAM (for small 1B-3B models)
- 8GB+ RAM (for 7B models)
- 16GB+ RAM (for 13B+ models)
- 2GB+ free disk space per model

### Recommended Setup
- 16GB+ RAM for comfortable usage with 7B models
- SSD storage for faster model loading
- Multi-core CPU for better throughput
- NVIDIA GPU for finetuning (optional but recommended)

## Performance Optimization

### Model Selection
1. Choose appropriate size: Start with 7B models for best speed/quality balance
2. Use quantized models: Q4_K_M offers good performance with reduced memory
3. Consider context length: Shorter contexts process faster

### Hardware Optimization
For NVIDIA GPU acceleration:
```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python --force-reinstall
```

### Memory Management
- Close unnecessary applications before running large models
- Use swap space if physical RAM is limited
- Monitor memory usage during benchmarking

## Troubleshooting

### Common Installation Issues

**"llama-cpp-python not installed"**
```bash
pip install llama-cpp-python
```

If issues persist, try:
```bash
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Permission errors with pip install**
```bash
pip install -e . --user
```

or use virtual environment:
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -e .
```

### Runtime Issues

**Out of memory errors**
- Try smaller models (7B instead of 13B)
- Use more aggressive quantization (Q4_0 instead of Q5_K_M)
- Reduce context size in the code
- Add swap space: `sudo swapon /swapfile`

**Slow performance**
- Verify CPU utilization with `top` or Task Manager
- Ensure model file is on fast storage (SSD)
- Check for background processes consuming resources
- Consider GPU acceleration for supported hardware

**Model download failures**
- Check internet connection and HuggingFace availability
- Verify model repository exists and is public
- Try downloading smaller test model first
- Use `--url` method if `--repo-id` fails

### Finetuning Issues

**CUDA out of memory during training**
Reduce batch size:
```bash
PARAMETER batch_size 1
```

Enable gradient accumulation:
```bash
PARAMETER gradient_accumulation_steps 4
```

Use LoRA for efficient training:
```bash
PARAMETER lora true
PARAMETER load_in_4bit true
```

**Slow training on CPU**
Use smaller models for CPU training:
```bash
FROM microsoft/DialoGPT-small
```

Reduce sequence length:
```bash
PARAMETER max_length 256
```

Use minimal training:
```bash
PARAMETER epochs 1
```

### Development Issues

**Import errors after installation**
```bash
which python
pip list | grep infernal
```

Reinstall in development mode:
```bash
pip uninstall infernal
pip install -e .
```

**Command not found after pip install**
```bash
pip show infernal
```

Add pip install directory to PATH if needed.

## Project Structure

```
infernal/
├── infernal.py                # Main application code
├── requirements.txt           # Dependencies for direct installation
├── pyproject.toml             # Modern Python project configuration
├── README.md                  # This documentation
├── tests/                     # Test suite
│   ├── test_benchmark.py
│   ├── test_cli.py
│   ├── test_inference.py
│   ├── test_model_management.py
│   └── test_integration.py
├── Modelfile                  # Example Modelfile for finetuning
└── models/                    # Model storage (created automatically)
    └── config.json            # Model configuration
```

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Update documentation if needed
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - High-performance inference engine
- [Hugging Face](https://huggingface.co/) - Model hosting and distribution platform
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output library
- [Click](https://click.palletsprojects.com/) - Command-line interface framework
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-tuning library

---

**Unleash the power of local LLMs with Infernal - inference, benchmarking, and finetuning made simple.**
