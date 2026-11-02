"""Checkpoint durability, request fidelity, and benchmark-backed evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from utils import LogWriter, compact_trace
import evaluate


class RunLoggingTests(unittest.TestCase):
    def setUp(self):
        printer = patch("builtins.print")
        printer.start()
        self.addCleanup(printer.stop)

    def setup_log_writer(self, directory, count=3):
        location = {'name': 'Test Library', 'address': 'Main Street', 'lat': 40.4, 'lon': -86.9}
        tasks = [{'task_id': str(i), 'operation': 'geocode', 'prompt': 'Find Test Library',
                  'answer': {'location': location}} for i in range(count)]
        source = Path(directory) / 'tasks.json'
        source.write_text(json.dumps(tasks))
        args = SimpleNamespace(output=str(Path(directory) / 'run.json'), terminal_log=None, tasks=[str(source)],
                               model='test', provider='openrouter', mcp_env=['SECRET=test-secret-value'])
        return LogWriter(args, 'baseline', tasks, '<role>test</role>', [], {'temperature': 0}), location

    def records(self, task_id, location):
        events = [{'type': 'llm_call', 'sequence': 1, 'turn': 1,
                   'request': {'model': 'test', 'messages': [], 'temperature': 0},
                   'response': {'id': 'response-1', 'usage': {'reasoning_tokens': 3}}}]
        return {
            'task_id': task_id, 'status': 'completed',
            'output': {'answer': {'location': location}, 'metrics': {
                'latency_ms': 10, 'cost_usd': .001, 'zero_hit': False, 'json_valid': True,
                'latency_breakdown_ms': {'llm_call_latency_ms': 9}, 'turn_count': 1,
                'prompt_tokens': 12, 'completion_tokens': 5, 'cached_tokens': 0}},
            'trace': compact_trace([{'role': 'user', 'content': 'test'}], events, {}),
        }

    def test_report_does_not_treat_missing_cost_as_zero(self):
        row = dict(task_id='test', correct=True, err_m=None, latency_s=1,
                   cost_usd=None, tool_calls=None, turns=1, grounded=None,
                   n_tool_errors=None, zero_hit=False)
        for costs, expected in [([None], 'n/a (0/1'), ([0], '$0.00000'),
                                ([None, .01], 'n/a (1/2')]:
            with self.subTest(costs=costs), patch('builtins.print') as printer:
                evaluate.print_report('geocode', [{**row, 'cost_usd': c} for c in costs])
                output = '\n'.join(str(call.args[0]) for call in printer.call_args_list)
                self.assertIn(expected, output)

    def test_report_handles_mixed_baseline_and_mcp_grounding(self):
        base = dict(task_id='test', correct=True, err_m=None, latency_s=1,
                    cost_usd=0, tool_calls=None, turns=1, n_tool_errors=None, zero_hit=False)
        rows = [{**base, 'grounded': None, 'correct_grounded': None},
                {**base, 'grounded': True, 'correct_grounded': True}]
        with patch('builtins.print') as printer:
            evaluate.print_report('geocode', rows)
        self.assertTrue(any('correct+grounded : 1/1' in str(call.args[0]) for call in printer.call_args_list))

    def test_checkpoint_schedule_and_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [0]
            with patch('utils.time.monotonic', side_effect=lambda: clock[0]):
                log_writer, location = self.setup_log_writer(directory, 15)
                with self.assertRaises(KeyboardInterrupt):
                    with log_writer:
                        log_writer.record(0, self.records('0', location))
                        self.assertEqual(json.loads(log_writer.path.read_text())['run']['completed_tasks'], 1)
                        log_writer.record(1, self.records('1', location))
                        self.assertEqual(json.loads(log_writer.path.read_text())['run']['completed_tasks'], 1)
                        clock[0] = 6
                        log_writer.checkpoint()
                        self.assertEqual(json.loads(log_writer.path.read_text())['run']['completed_tasks'], 2)
                        for i in range(2, 12):
                            log_writer.record(i, self.records(str(i), location))
                        self.assertEqual(json.loads(log_writer.path.read_text())['run']['completed_tasks'], 12)
                        raise KeyboardInterrupt()
                document = json.loads(log_writer.path.read_text())
                self.assertEqual(document['run']['status'], 'interrupted')
                self.assertEqual(document['tasks'][-1]['status'], 'pending')
                self.assertNotIn('test-secret-value', json.dumps(document))
                self.assertNotIn('mcp', document['config'])

    def test_failed_replace_preserves_last_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            log_writer, _ = self.setup_log_writer(directory)
            log_writer.checkpoint(force=True)
            previous = log_writer.path.read_bytes()
            log_writer.document['run']['status'] = 'changed'
            log_writer.dirty = True
            with patch('utils.os.replace', side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError):
                    log_writer.checkpoint(force=True)
            self.assertEqual(log_writer.path.read_bytes(), previous)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['run.json', 'tasks.json'])

    def test_evaluation_uses_benchmark_and_skips_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            log_writer, location = self.setup_log_writer(directory, 2)
            record = self.records('0', location)
            with log_writer:
                log_writer.record(0, record)
            rows = evaluate.evaluate(None, [str(log_writer.path)])['geocode']
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]['correct'])
            self.assertTrue(rows[0]['json_valid'])
            task = json.loads(log_writer.path.read_text(encoding='utf-8'))['tasks'][0]
            self.assertEqual(task['trace']['events'][0]['response'], record['trace']['events'][0]['response'])
            self.assertEqual(task['output']['answer'], record['output']['answer'])

    def test_evaluation_can_override_missing_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            log_writer, location = self.setup_log_writer(directory, 1)
            with log_writer:
                log_writer.record(0, self.records('0', location))
            source = Path(log_writer.document['config']['tasks'][0])
            replacement = Path(directory) / 'replacement.json'
            replacement.write_bytes(source.read_bytes())
            source.unlink()
            self.assertTrue(evaluate.evaluate(str(replacement), [str(log_writer.path)])['geocode'][0]['correct'])
            tasks = json.loads(replacement.read_text())
            tasks[0]['answer']['location'].update(lat=0, lon=0)
            replacement.write_text(json.dumps(tasks))
            self.assertFalse(evaluate.evaluate(str(replacement), [str(log_writer.path)])['geocode'][0]['correct'])

    def test_capture_console_log_captures_subprocess_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'run.log'
            script = '''from utils import capture_console_log
import subprocess,sys
with capture_console_log(sys.argv[1]):
 print("stdout message",flush=True)
 print("stderr message",file=sys.stderr,flush=True)
 subprocess.run([sys.executable,"-c","import sys; print('child message',file=sys.stderr)"],check=True)
print("after backup",flush=True)
'''
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')}
            result = subprocess.run([sys.executable, '-c', script, str(log)], env=env, capture_output=True, text=True, check=True)
            text = log.read_text()
            for message in ['stdout message', 'stderr message', 'child message']:
                self.assertEqual(text.count(message), 1)
                self.assertIn(message, result.stdout)
            self.assertNotIn('after backup', text)
            # Existing shell-style redirection to run.log must not duplicate or truncate output.
            with log.open('w') as target:
                subprocess.run([sys.executable, '-c', script, str(log)], env=env, stdout=target, stderr=target, check=True)
            self.assertEqual(log.read_text().count('stdout message'), 1)

    def test_capture_console_log_drains_output_and_restores_streams_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'run.log'
            script = """from utils import capture_console_log
import sys
try:
 with capture_console_log(sys.argv[1]):
  print('x' * 200000, flush=True)
  raise RuntimeError('test failure')
except RuntimeError:
 print('restored stdout', flush=True)
 print('restored stderr', file=sys.stderr, flush=True)
"""
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')}
            result = subprocess.run([sys.executable, '-c', script, str(log)], env=env,
                                    capture_output=True, text=True, check=True, timeout=15)
            captured = log.read_text()
            self.assertIn('x' * 200000, captured)
            self.assertIn('RuntimeError: test failure', captured)
            self.assertTrue(result.stdout.startswith(captured))
            self.assertNotIn('restored stdout', captured)
            self.assertIn('restored stdout', result.stdout)
            self.assertEqual(result.stderr.strip(), 'restored stderr')

    def test_capture_console_log_handles_console_failures(self):
        script = """from utils import capture_console_log
from pathlib import Path
from unittest.mock import patch
import os,sys
saved = os.dup(1)
original_open = Path.open

def open_file(self, mode='r', *args, **kwargs):
 if mode == 'rb' and sys.argv[2] != 'broken':
  raise OSError('mirror read failed')
 return original_open(self, mode, *args, **kwargs)

if sys.argv[2] == 'broken':
 reader, writer = os.pipe()
 os.close(reader)
 os.dup2(writer, 1)
 os.close(writer)
try:
 with patch.object(Path, 'open', open_file):
  with capture_console_log(sys.argv[1]):
   print('saved output', flush=True)
   if sys.argv[2] == 'original_error':
    raise ValueError('original failure')
except RuntimeError as error:
 assert sys.argv[2] == 'mirror_error'
 assert isinstance(error.__cause__, OSError)
except ValueError as error:
 assert sys.argv[2] == 'original_error' and str(error) == 'original failure'
else:
 assert sys.argv[2] == 'broken'
finally:
 os.dup2(saved, 1)
 os.close(saved)
"""
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('broken', 'mirror_error', 'original_error'):
                with self.subTest(mode=mode):
                    log = Path(directory) / (mode + '.log')
                    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')}
                    result = subprocess.run([sys.executable, '-c', script, str(log), mode], env=env,
                                            capture_output=True, text=True, check=True, timeout=15)
                    self.assertIn('saved output', log.read_text())
                    self.assertNotIn('Exception in thread', result.stderr)
                    if mode == 'original_error':
                        self.assertIn('Console log mirroring failed', result.stderr)

    def test_capture_console_log_exits_and_preserves_background_output(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'run.log'
            script = '''from utils import capture_console_log
import subprocess,sys
child = None
try:
 with capture_console_log(sys.argv[1]):
  print("saved before cleanup",flush=True)
  child = subprocess.Popen([sys.executable,"-c","import sys; sys.stdin.readline(); print('late child output',flush=True)"],stdin=subprocess.PIPE,text=True)
 print("cleanup completed",flush=True)
 child.communicate("go\\n",timeout=5)
 assert child.returncode == 0
finally:
 if child is not None:
  child.terminate()
  child.wait()
'''
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')}
            result = subprocess.run([sys.executable, '-c', script, str(log)], env=env,
                                    capture_output=True, text=True, check=True, timeout=15)
            self.assertIn('saved before cleanup', log.read_text())
            self.assertIn('cleanup completed', result.stdout)
            self.assertNotIn('cleanup completed', log.read_text())
            self.assertIn('late child output', log.read_text())
            self.assertNotIn('late child output', result.stdout)


if __name__ == '__main__':
    unittest.main()
