import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from infernal import InfernalLLM

class TestIntegration:
    """Integration tests that test multiple components working together"""
    
    @pytest.fixture
    def temp_infernal(self, tmp_path):
        """Create a temporary Infernal instance for integration testing"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        return InfernalLLM(str(models_dir))

    def test_full_model_lifecycle(self, temp_infernal):
        """Test complete model lifecycle: download, list, run, remove"""
        
        # Mock the actual download but simulate the config update
        with patch('infernal.huggingface_hub.hf_hub_download') as mock_download:
            mock_download.return_value = str(temp_infernal.models_dir / "downloaded.gguf")
            
            # Create a fake model file
            model_file = temp_infernal.models_dir / "test_repo.gguf"
            model_file.touch()
            
            # Test download
            result_path = temp_infernal.download_from_huggingface("test/test_repo", "model.gguf")
            assert result_path.exists()
            
            # Verify config was updated
            assert "test_repo" in temp_infernal.config["models"]
            assert temp_infernal.config["models"]["test_repo"]["filename"] == "test_repo.gguf"
            
            # Test model path retrieval
            model_path = temp_infernal.get_model_path("test_repo")
            assert model_path == model_file
            
            # Test removal
            temp_infernal.remove_model("test_repo")
            assert not model_file.exists()
            assert "test_repo" not in temp_infernal.config["models"]

    def test_config_persistence(self, temp_infernal):
        """Test that configuration persists across instances"""
        # Add a model to config
        temp_infernal.config["models"]["persistent_model"] = {
            "filename": "persistent.gguf",
            "repo_id": "test/persistent",
            "size": 1000000
        }
        temp_infernal.save_config()
        
        # Create new instance with same directory
        new_infernal = InfernalLLM(str(temp_infernal.models_dir))
        
        # Verify config was loaded
        assert "persistent_model" in new_infernal.config["models"]
        assert new_infernal.config["models"]["persistent_model"]["repo_id"] == "test/persistent"

    @patch('infernal.Llama')
    def test_inference_with_config_settings(self, mock_llama_class, temp_infernal):
        """Test that inference uses configuration settings"""
        # Setup model
        temp_infernal.config["models"]["test_model"] = {"filename": "test.gguf"}
        model_file = temp_infernal.models_dir / "test.gguf"
        model_file.touch()
        
        # Configure custom settings
        temp_infernal.config["settings"] = {
            "max_tokens": 1024,
            "temperature": 0.5,
            "top_p": 0.8
        }
        
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm
        mock_llm.return_value = {"choices": [{"text": "Test response"}]}
        
        # Run inference
        temp_infernal.run_inference("test_model", "Test prompt")
        
        # Verify settings were used
        mock_llm.assert_called_with(
            "Test prompt",
            max_tokens=1024,
            temperature=0.5,
            top_p=0.8,
            stop=["User:", "\n\n"]
        )

    def test_url_parsing_and_download(self, temp_infernal):
        """Test URL parsing and download integration"""
        test_url = "https://huggingface.co/test/repo/resolve/main/model.gguf"
        
        with patch.object(temp_infernal, 'pull_model') as mock_pull:
            temp_infernal.pull_from_url(test_url)
            mock_pull.assert_called_once_with("test/repo", "model.gguf")

    @patch('infernal.Llama')
    def test_benchmark_integration(self, mock_llama_class, temp_infernal):
        """Test benchmark integration with real-like data flow"""
        # Setup model
        temp_infernal.config["models"]["bench_model"] = {"filename": "bench.gguf"}
        model_file = temp_infernal.models_dir / "bench.gguf"
        model_file.touch()
        
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm
        
        # Simulate streaming response
        mock_llm.return_value = iter([
            {"choices": [{"text": "First "}]},
            {"choices": [{"text": "response "}]},
            {"choices": [{"text": "complete."}]}
        ])
        
        # Mock tokenization
        mock_llm.tokenize.side_effect = lambda x, *args: list(range(len(x)))
        
        with patch('time.perf_counter', side_effect=[0, 0.5, 2.0]), \
             patch('psutil.Process') as mock_process:
            
            mock_process.return_value.memory_info.return_value.rss = 500 * 1024 * 1024
            
            # Run benchmark
            temp_infernal.benchmark_model("bench_model", ["Test prompt"], repeat=1)
            
            # Verify llama was called with streaming
            mock_llm.assert_called_with(
                "Test prompt",
                max_tokens=temp_infernal.config["settings"]["max_tokens"],
                temperature=temp_infernal.config["settings"]["temperature"],
                top_p=temp_infernal.config["settings"]["top_p"],
                stop=["User:", "\n\n"],
                stream=True
            )
