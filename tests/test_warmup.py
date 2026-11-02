"""Warm-up phase separation, durability and evaluator isolation (no API calls)."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import evaluate
from utils import LogWriter, select_tasks
import run_baseline as baseline
import run_mcp as mcp

class WarmupTests(unittest.IsolatedAsyncioTestCase):
    def writer(self, directory, count, tasks):
        source = Path(directory) / 'tasks.json'
        source.write_text(json.dumps(tasks))
        args = SimpleNamespace(output=str(Path(directory) / 'run.json'), terminal_log=None,
                               tasks=[str(source)], warmup_runs=count, model="test", provider="openrouter")
        return LogWriter(args, 'baseline', tasks, 'system', [], {})

    def test_negative_count_rejected_before_loading_tasks(self):
        with self.assertRaisesRegex(ValueError, "--warmup-runs"):
            select_tasks(SimpleNamespace(warmup_runs=-1))

    async def test_invalid_options_fail_before_client_creation(self):
        cases = [(baseline, "--num-workers", "0"),
                 (baseline, "--num-workers", "-1"),
                 (baseline, "--warmup-runs", "-1"),
                 (mcp, "--warmup-runs", "-1")]
        for module, option, value in cases:
            with self.subTest(runner=module.__name__, option=option, value=value):
                argv = ["runner", "--tasks", "missing-tasks.json", "--output", "unused.json", option, value]
                with patch.object(sys, "argv", argv), patch.object(module, "build_openai_client") as build:
                    with self.assertRaisesRegex(ValueError, "num_workers|warmup-runs"):
                        await module.main()
                    build.assert_not_called()

    def result(self, task, cost=.01):
        return dict(task_id=task['task_id'], status='completed',
                    output={'answer': {'location': task['answer']['location']}, 'metrics': {
                        'cost_usd': cost, 'latency_ms': 1, 'turn_count': 1,
                        'prompt_tokens': 10, 'completion_tokens': 5, 'cached_tokens': 0,
                        'json_valid': True, 'zero_hit': False}}, trace={'transcript': [], 'events': []})

    async def test_repeats_first_selected_task_and_keeps_scoring_separate(self):
        loc = {'name': 'Test', 'lat': 40., 'lon': -86.}
        tasks = [{'task_id': 'selected', 'operation': 'geocode', 'prompt': 'find Test',
                  'answer': {'location': loc}}]
        seen = []
        async def worker(i, task):
            seen.append(copy.deepcopy(task))
            result = self.result(task)
            task['prompt'] = 'mutated by worker'
            return result
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            writer = self.writer(directory, 2, tasks)
            with writer:
                self.assertIsNone(writer.document['run']['started_at'])
                await writer.warmup(tasks, worker)
                self.assertEqual(writer.document['run']['completed_tasks'], 0)
                self.assertEqual(tasks[0]['prompt'], 'find Test')
                self.assertEqual([t['prompt'] for t in seen], ['find Test', 'find Test'])
                writer.record(0, self.result(tasks[0], .03))
            d = json.loads(writer.path.read_text())
            self.assertEqual(len(d['warmup_tasks']), 2)
            self.assertEqual(d['run']['total_tasks'], 1)
            self.assertEqual(d['run']['completed_tasks'], 1)
            self.assertIsNotNone(d['run']['started_at'])
            self.assertNotIn('benchmark_started_at', d['run'])
            self.assertNotIn('summary', d)
            self.assertEqual(sum(t['output']['metrics']['cost_usd'] for t in d['warmup_tasks']), .02)
            self.assertEqual(d['tasks'][0]['output']['metrics']['cost_usd'], .03)
            self.assertNotIn('answer', d['warmup_tasks'][0]['input'])
            scores = evaluate.evaluate(None, [str(writer.path)])
            self.assertEqual(len(scores['geocode']), 1)
            self.assertTrue(scores['geocode'][0]['correct'])

    async def test_disable_empty_interruption_and_unknown_cost(self):
        task = {'task_id': 'a', 'answer': {'location': {'lat': 1, 'lon': 2}}}
        for count, tasks in [(0, [task]), (2, [])]:
            with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
                writer = self.writer(directory, count, tasks)
                with writer:
                    async def unexpected(*args):
                        self.fail('No warm-up should run')
                    await writer.warmup(tasks, unexpected)
                self.assertEqual(writer.document['warmup_tasks'], [])
                self.assertIsNotNone(writer.document['run']['started_at'])
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            writer = self.writer(directory, 2, [task])
            async def worker(i, t):
                if i:
                    raise RuntimeError('interrupted warmup')
                return self.result(t, None)
            with self.assertRaisesRegex(RuntimeError, 'interrupted warmup'):
                with writer:
                    await writer.warmup([task], worker)
            d = json.loads(writer.path.read_text())
            self.assertEqual(d['warmup_tasks'][0]['status'], 'completed')
            self.assertEqual(d['warmup_tasks'][1]['status'], 'pending')
            self.assertIsNone(d['run']['started_at'])
            self.assertEqual(d['run']['completed_tasks'], 0)
            self.assertIsNone(d['warmup_tasks'][0]['output']['metrics']['cost_usd'])
