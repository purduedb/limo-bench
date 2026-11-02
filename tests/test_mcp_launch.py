"""The Make entry point builds before serving without polluting MCP stdout."""
from pathlib import Path
import subprocess
import tempfile
import os
import shlex
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MCPLaunchTests(unittest.TestCase):
    def launch(self, build):
        return subprocess.run(
            ['make', '--silent', '--no-print-directory', 'mcp_serve',
             f'MCP_BUILD={build}', 'MCP_COMMAND=printf', 'MCP_ARGS=protocol', 'MCP_ENV='],
            cwd=ROOT, capture_output=True, text=True,
        )

    def test_build_output_is_stderr_and_server_output_is_stdout(self):
        result = self.launch('printf build-log')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'protocol')
        self.assertEqual(result.stderr, 'build-log')

    def test_build_failure_prevents_serving(self):
        result = self.launch('false')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')

    def test_vendor_without_build_still_serves(self):
        result = self.launch('')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'protocol')

    def test_dotenv_quoted_values_reach_application_without_literal_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / '.env').write_text('TEST_API_URL="https://example.invalid/api/v1"\n')
            code = 'import os; from dotenv import load_dotenv; load_dotenv(".env"); print(os.environ["TEST_API_URL"])'
            probe = folder / 'probe.mk'
            probe.write_text(f'probe:\n\t@{shlex.quote(sys.executable)} -c {shlex.quote(code)}\n')
            env = {k: v for k, v in os.environ.items() if k != 'TEST_API_URL'}
            result = subprocess.run(['make', '--silent', '-f', str(ROOT / 'Makefile'), '-f', str(probe), 'probe'],
                                    cwd=folder, env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), 'https://example.invalid/api/v1')
