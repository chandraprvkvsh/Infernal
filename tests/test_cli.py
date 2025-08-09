import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from infernal import cli, InfernalLLM

@pytest.fixture
def runner():
    """Click test runner"""
    return CliRunner()

@pytest.fixture
def mock_infernal():
    """Mock InfernalLLM instance"""
    return MagicMock(spec=InfernalLLM)

class TestCLI:
    def test_cli_version(self, runner):
        """Test CLI version command"""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '1.0.0' in result.output

    def test_cli_help(self, runner):
        """Test CLI help command"""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Infernal' in result.output
        assert 'pull' in result.output
        assert 'run' in result.output

    @patch('infernal.InfernalLLM')
    def test_pull_with_url(self, mock_class, runner, mock_infernal):
        """Test pull command with URL"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, [
            'pull',
            '--url',
            'https://huggingface.co/test/repo/resolve/main/model.gguf'
        ])
        assert result.exit_code == 0
        mock_infernal.download_model.assert_called_once()

    @patch('infernal.InfernalLLM')
    def test_pull_with_repo_id(self, mock_class, runner, mock_infernal):
        """Test pull command with repo-id and filename"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, [
            'pull',
            '--repo-id', 'test/repo',
            '--filename', 'model.gguf'
        ])
        assert result.exit_code == 0
        mock_infernal.download_from_huggingface.assert_called_once_with('test/repo', 'model.gguf')

    @patch('infernal.InfernalLLM')
    def test_pull_no_arguments(self, mock_class, runner, mock_infernal):
        """Test pull command without arguments"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, ['pull'])
        assert result.exit_code == 0
        assert 'Please provide either --url or both --repo-id and --filename' in result.output

    @patch('infernal.InfernalLLM')
    def test_list_command(self, mock_class, runner, mock_infernal):
        """Test list command"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, ['list'])
        assert result.exit_code == 0
        mock_infernal.list_models.assert_called_once()

    @patch('infernal.InfernalLLM')
    def test_run_command_with_prompt(self, mock_class, runner, mock_infernal):
        """Test run command with prompt"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, [
            'run', 'test_model',
            '--prompt', 'Hello world'
        ])
        assert result.exit_code == 0
        mock_infernal.run_model.assert_called_once_with(
            'test_model', 'Hello world', False
        )

    @patch('infernal.InfernalLLM')
    def test_run_command_interactive(self, mock_class, runner, mock_infernal):
        """Test run command in interactive mode"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, [
            'run', 'test_model',
            '--interactive'
        ])
        assert result.exit_code == 0
        mock_infernal.run_model.assert_called_once_with(
            'test_model', None, True
        )

    @patch('infernal.InfernalLLM')
    def test_remove_command(self, mock_class, runner, mock_infernal):
        """Test remove command"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, ['remove', 'test_model'])
        assert result.exit_code == 0
        mock_infernal.remove_model.assert_called_once_with('test_model')

    @patch('infernal.InfernalLLM')
    def test_benchmark_command_with_prompt(self, mock_class, runner, mock_infernal):
        """Test benchmark command with prompt"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, [
            'benchmark', 'test_model',
            '--prompt', 'Test prompt',
            '--repeat', '3'
        ])
        assert result.exit_code == 0
        mock_infernal.benchmark_model.assert_called_once()

    @patch('infernal.InfernalLLM')
    def test_benchmark_command_no_prompt(self, mock_class, runner, mock_infernal):
        """Test benchmark command without prompt - Fixed assertion"""
        mock_class.return_value = mock_infernal
        result = runner.invoke(cli, ['benchmark', 'test_model'])
        assert result.exit_code == 0
        # Fixed: Use the exact text from your code
        assert 'Please provide --prompt for benchmarking' in result.output

    @patch('infernal.InfernalLLM')
    def test_models_dir_option(self, mock_class, runner):
        """Test custom models directory option"""
        result = runner.invoke(cli, ['--models-dir', '/custom/path', 'list'])
        assert result.exit_code == 0
        mock_class.assert_called_once_with('/custom/path')
