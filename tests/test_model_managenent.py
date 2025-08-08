import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from infernal import InfernalLLM

@pytest.fixture
def temp_models_dir(tmp_path):
    """Create a temporary models directory"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir

@pytest.fixture
def infernal_with_config(temp_models_dir):
    """Create InfernalLLM instance with test configuration"""
    infernal = InfernalLLM(str(temp_models_dir))
    infernal.config = {
        "models": {
            "test_model": {
                "filename": "test_model.gguf",
                "repo_id": "test/repo",
                "original_filename": "original.gguf",
                "size": 1024000
            }
        },
        "default_model": None,
        "settings": {
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    return infernal

class TestInfernalLLM:
    def test_init_creates_models_dir(self, tmp_path):
        """Test that initialization creates models directory"""
        models_dir = tmp_path / "test_models"
        infernal = InfernalLLM(str(models_dir))
        
        assert models_dir.exists()
        assert infernal.models_dir == models_dir

    def test_init_default_models_dir(self):
        """Test default models directory creation"""
        infernal = InfernalLLM()
        assert infernal.models_dir.name in ["models", ".infernal-models"]

    def test_load_config_creates_default(self, temp_models_dir):
        """Test that load_config creates default configuration"""
        infernal = InfernalLLM(str(temp_models_dir))
        
        assert "models" in infernal.config
        assert "settings" in infernal.config
        assert infernal.config["settings"]["max_tokens"] == 2048

    def test_load_config_from_existing_file(self, temp_models_dir):
        """Test loading configuration from existing file"""
        config_data = {
            "models": {"existing_model": {"filename": "existing.gguf"}},
            "settings": {"max_tokens": 1024}
        }
        
        config_file = temp_models_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        infernal = InfernalLLM(str(temp_models_dir))
        
        assert infernal.config["models"]["existing_model"]["filename"] == "existing.gguf"
        assert infernal.config["settings"]["max_tokens"] == 1024

    def test_save_config(self, infernal_with_config):
        """Test saving configuration to file"""
        infernal_with_config.save_config()
        
        config_file = infernal_with_config.config_file
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert "test_model" in saved_config["models"]
        assert saved_config["settings"]["max_tokens"] == 2048

    def test_get_model_path_existing(self, infernal_with_config):
        """Test getting path for existing model"""
        model_file = infernal_with_config.models_dir / "test_model.gguf"
        model_file.touch()
        
        path = infernal_with_config.get_model_path("test_model")
        assert path == model_file

    def test_get_model_path_nonexistent(self, infernal_with_config):
        """Test getting path for non-existent model"""
        path = infernal_with_config.get_model_path("nonexistent")
        assert path is None

    def test_get_model_files(self, infernal_with_config):
        """Test getting all GGUF model files"""
        # Create test files
        (infernal_with_config.models_dir / "model1.gguf").touch()
        (infernal_with_config.models_dir / "model2.gguf").touch()
        (infernal_with_config.models_dir / "not_gguf.txt").touch()
        
        files = infernal_with_config.get_model_files()
        
        assert len(files) == 2
        assert all(f.suffix == ".gguf" for f in files)

    def test_remove_model_success(self, infernal_with_config, capsys):
        """Test successful model removal"""
        model_file = infernal_with_config.models_dir / "test_model.gguf"
        model_file.touch()
        
        infernal_with_config.remove_model("test_model")
        
        assert not model_file.exists()
        assert "test_model" not in infernal_with_config.config["models"]
        
        captured = capsys.readouterr()
        assert "Removed test_model" in captured.out

    def test_remove_model_not_found(self, infernal_with_config, capsys):
        """Test removing non-existent model"""
        infernal_with_config.remove_model("nonexistent")
        
        captured = capsys.readouterr()
        assert "Model nonexistent not found" in captured.out

    def test_list_models_empty(self, temp_models_dir, capsys):
        """Test listing models when none exist"""
        infernal = InfernalLLM(str(temp_models_dir))
        infernal.list_models()
        
        captured = capsys.readouterr()
        assert "No models found" in captured.out

    def test_list_models_with_models(self, infernal_with_config, capsys):
        """Test listing models when models exist"""
        model_file = infernal_with_config.models_dir / "test_model.gguf"
        model_file.write_bytes(b"0" * 1024)  # 1KB file
        
        infernal_with_config.list_models()
        
        captured = capsys.readouterr()
        assert "Available Models:" in captured.out
        assert "test_model" in captured.out


class TestModelDownload:
    @patch('infernal.huggingface_hub.hf_hub_download')
    def test_pull_model_success(self, mock_download, infernal_with_config, capsys):
        """Test successful model download"""
        mock_download.return_value = str(infernal_with_config.models_dir / "downloaded.gguf")
        
        result = infernal_with_config.pull_model("test/repo", "model.gguf")
        
        assert result.name == "repo.gguf"
        mock_download.assert_called_once()
        
        captured = capsys.readouterr()
        assert "Successfully downloaded" in captured.out

    @patch('infernal.huggingface_hub.hf_hub_download')
    def test_pull_model_already_exists(self, mock_download, infernal_with_config, capsys):
        """Test downloading model that already exists"""
        model_file = infernal_with_config.models_dir / "repo.gguf"
        model_file.touch()
        
        result = infernal_with_config.pull_model("test/repo", "model.gguf")
        
        assert result == model_file
        mock_download.assert_not_called()
        
        captured = capsys.readouterr()
        assert "already exists" in captured.out

    @patch('infernal.huggingface_hub.hf_hub_download')
    def test_pull_model_download_failure(self, mock_download, infernal_with_config):
        """Test handling download failure"""
        mock_download.side_effect = Exception("Download failed")
        
        with pytest.raises(Exception, match="Download failed"):
            infernal_with_config.pull_model("test/repo", "model.gguf")

    def test_pull_from_url_valid(self, infernal_with_config):
        """Test extracting repo info from valid HuggingFace URL"""
        url = "https://huggingface.co/test/repo/resolve/main/model.gguf"
        
        with patch.object(infernal_with_config, 'pull_model') as mock_pull:
            infernal_with_config.pull_from_url(url)
            mock_pull.assert_called_once_with("test/repo", "model.gguf")

    def test_pull_from_url_invalid(self, infernal_with_config):
        """Test handling invalid URL"""
        url = "https://example.com/model.gguf"
        
        with pytest.raises(ValueError, match="Only Hugging Face URLs supported"):
            infernal_with_config.pull_from_url(url)
