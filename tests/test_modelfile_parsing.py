import pytest
import tempfile
from pathlib import Path
from infernal import parse_modelfile

class TestModelfileParsing:
    def test_parse_basic_modelfile(self, tmp_path):
        """Test parsing a basic Modelfile"""
        modelfile_content = """
FROM meta-llama/Llama-3.2-1B-Instruct
PARAMETER learning_rate 3e-5
PARAMETER epochs 3
SYSTEM You are a helpful assistant.
MESSAGE user Hello
MESSAGE assistant Hi there!
HF_TOKEN test_token
"""
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['FROM'] == 'meta-llama/Llama-3.2-1B-Instruct'
        assert config['PARAMETER']['learning_rate'] == '3e-5'
        assert config['PARAMETER']['epochs'] == '3'
        assert config['SYSTEM'] == 'You are a helpful assistant.'
        assert config['HF_TOKEN'] == 'test_token'
        assert len(config['MESSAGES']) == 2
        assert config['MESSAGES'][0]['role'] == 'user'
        assert config['MESSAGES'][0]['content'] == 'Hello'

    def test_parse_modelfile_with_template(self, tmp_path):
        """Test parsing Modelfile with template"""
        modelfile_content = """
FROM test/model
TEMPLATE User: {{ .Prompt }}
Assistant: {{ .Response }}
"""
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['FROM'] == 'test/model'
        assert 'User: {{ .Prompt }}' in config['TEMPLATE']
        assert 'Assistant: {{ .Response }}' in config['TEMPLATE']

    def test_parse_modelfile_with_comments(self, tmp_path):
        """Test parsing Modelfile with comments and empty lines"""
        modelfile_content = """
# This is a comment
FROM test/model

# Another comment
PARAMETER learning_rate 2e-5
# More comments

SYSTEM You are helpful.
"""
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['FROM'] == 'test/model'
        assert config['PARAMETER']['learning_rate'] == '2e-5'
        assert config['SYSTEM'] == 'You are helpful.'

    def test_parse_empty_modelfile(self, tmp_path):
        """Test parsing empty Modelfile"""
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text("")
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['FROM'] is None
        assert config['SYSTEM'] is None
        assert len(config['PARAMETER']) == 0
        assert len(config['MESSAGES']) == 0

    def test_parse_modelfile_multiline_template(self, tmp_path):
        """Test parsing Modelfile with multiline template"""
        modelfile_content = '''
FROM test/model
TEMPLATE System: {{ .System }}
User: {{ .Prompt }}
Assistant: {{ .Response }}
'''
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['FROM'] == 'test/model'
        expected_template = '''System: {{ .System }}
User: {{ .Prompt }}
Assistant: {{ .Response }}'''
        assert config['TEMPLATE'] == expected_template

    def test_parse_modelfile_lora_parameters(self, tmp_path):
        """Test parsing Modelfile with LoRA parameters"""
        modelfile_content = """
FROM test/model
PARAMETER lora true
PARAMETER lora_r 16
PARAMETER lora_alpha 64
PARAMETER lora_dropout 0.1
PARAMETER lora_target_modules q_proj,v_proj,k_proj,o_proj
PARAMETER load_in_4bit true
"""
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['PARAMETER']['lora'] == 'true'
        assert config['PARAMETER']['lora_r'] == '16'
        assert config['PARAMETER']['lora_alpha'] == '64'
        assert config['PARAMETER']['lora_dropout'] == '0.1'
        assert config['PARAMETER']['lora_target_modules'] == 'q_proj,v_proj,k_proj,o_proj'
        assert config['PARAMETER']['load_in_4bit'] == 'true'

    def test_parse_modelfile_training_parameters(self, tmp_path):
        """Test parsing Modelfile with comprehensive training parameters"""
        modelfile_content = """
FROM meta-llama/Llama-3.2-1B-Instruct
PARAMETER device cuda
PARAMETER max_length 2048
PARAMETER learning_rate 2e-5
PARAMETER epochs 5
PARAMETER batch_size 4
PARAMETER weight_decay 0.01
PARAMETER warmup_steps 100
PARAMETER gradient_accumulation_steps 2
PARAMETER fp16 true
PARAMETER save_steps 500
PARAMETER logging_steps 10
PARAMETER lr_scheduler_type cosine
PARAMETER eval_steps 100
PARAMETER save_total_limit 3
PARAMETER seed 42
"""
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['PARAMETER']['device'] == 'cuda'
        assert config['PARAMETER']['max_length'] == '2048'
        assert config['PARAMETER']['learning_rate'] == '2e-5'
        assert config['PARAMETER']['epochs'] == '5'
        assert config['PARAMETER']['batch_size'] == '4'
        assert config['PARAMETER']['lr_scheduler_type'] == 'cosine'

    def test_parse_modelfile_multiple_messages(self, tmp_path):
        """Test parsing Modelfile with multiple message pairs"""
        modelfile_content = """
FROM test/model
SYSTEM You are a customer service bot.
MESSAGE user How do I return an item?
MESSAGE assistant You can return items within 30 days by visiting our returns page.
MESSAGE user What about refunds?
MESSAGE assistant Refunds are processed within 5-7 business days after we receive your return.
MESSAGE user Thank you
MESSAGE assistant You're welcome! Is there anything else I can help you with?
"""
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        assert config['SYSTEM'] == 'You are a customer service bot.'
        assert len(config['MESSAGES']) == 6
        assert config['MESSAGES'][0]['role'] == 'user'
        assert config['MESSAGES'][1]['role'] == 'assistant'
        assert 'return items' in config['MESSAGES'][1]['content']

    def test_parse_invalid_message_format(self, tmp_path):
        """Test parsing Modelfile with invalid message format"""
        modelfile_content = """
FROM test/model
MESSAGE invalid_format This should not parse
MESSAGE user This should parse
MESSAGE assistant This should also parse
"""
        
        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        config = parse_modelfile(str(modelfile_path))
        
        # Should only parse valid messages
        assert len(config['MESSAGES']) == 2
        assert config['MESSAGES'][0]['role'] == 'user'
        assert config['MESSAGES'][1]['role'] == 'assistant'
