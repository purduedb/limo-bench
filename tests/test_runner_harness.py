"""Offline regression checks for shared runner plumbing and MCP orchestration."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from openai.types.chat import ChatCompletion

import utils as api
import run_baseline as baseline
import run_mcp as mcp
from utils import LogWriter


class Call:
    id = 'call_test'
    function = SimpleNamespace(name='range_search', arguments='{"a":1}')

    def model_dump(self):
        return {'id': self.id, 'function': vars(self.function), 'type': 'function'}


def response(content='{}', calls=None, finish='stop'):
    return ChatCompletion.model_validate({
        'id': 'response-test', 'created': 1, 'object': 'chat.completion', 'model': 'test',
        'provider': 'test-provider',
        'choices': [{'index': 0, 'finish_reason': finish, 'message': {
            'role': 'assistant', 'content': content,
            'tool_calls': [call.model_dump() for call in calls or []],
            'reasoning_content': 'provider reasoning text'}}],
        'usage': {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17,
                  'completion_tokens_details': {'reasoning_tokens': 3}, 'cost': 0.001},
    })


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    def test_mcp_environment_inheritance_and_overrides(self):
        with patch.dict(os.environ, {'DB_NAME': 'inherited', 'DB_PASSWORD': 'test-password'}, clear=True):
            inherited = mcp.parse_env_vars([])
            self.assertEqual(inherited, dict(os.environ))
            self.assertIsNot(inherited, os.environ)
            overridden = mcp.parse_env_vars(['DB_NAME=selected', 'EXTRA=value=with=equals'])
            self.assertEqual(overridden['DB_NAME'], 'selected')
            self.assertEqual(overridden['DB_PASSWORD'], 'test-password')
            self.assertEqual(overridden['EXTRA'], 'value=with=equals')
            self.assertEqual(os.environ['DB_NAME'], 'inherited')
            with self.assertRaises(ValueError):
                mcp.parse_env_vars(['invalid'])

    async def test_coordinator_passes_finished_record_to_writer(self):
        record = {'status': 'completed', 'output': {'answer': {}}}
        saved = []

        async def worker(index, task):
            return record

        results = await api.run_tasks([{'task_id': 'test'}], worker,
                                      on_result=lambda index, payload: saved.append((index, payload)))
        self.assertEqual(results, [record])
        self.assertEqual(saved, [(0, record)])

    async def test_anthropic_compatibility_request(self):
        create = AsyncMock(return_value=response())
        client = SimpleNamespace(base_url='https://api.anthropic.com/v1/',
                                 chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        parameters = {'temperature': 0, 'seed': 42, 'max_tokens': 64,
                      'response_format': {'type': 'json_object'}, 'reasoning_effort': 'low',
                      'tool_choice': 'none'}
        events = []
        _, error, _ = await api.request_completion(client, 'claude-haiku-5-5', [], parameters, events)
        self.assertIsNone(error)
        sent = create.call_args.kwargs
        for key in ('temperature', 'seed', 'response_format', 'reasoning_effort', 'tool_choice'):
            self.assertNotIn(key, sent)
        self.assertEqual(sent['extra_body'], {'output_config': {'effort': 'low'}})
        self.assertEqual(events[0]['request'], sent)
        self.assertEqual(parameters['temperature'], 0)

    async def run_loop(self, final='answer', budget=20, base_url='https://example.invalid'):
        requests = []

        async def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            last = kwargs.get('tool_choice') == 'none'
            if last and final != 'tools':
                return response('{}' if final == 'answer' else '')
            return response(None, [Call()], 'tool_calls')

        client = SimpleNamespace(base_url=base_url, close=AsyncMock(), chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        tools = SimpleNamespace(call_tool=AsyncMock(return_value={'count': 1}))
        with patch.object(mcp, '_log'), patch.object(mcp, 'estimate_llm_cost', return_value=0):
            record = await mcp.agent_loop(client, tools, {'task_id': 'test', 'prompt': 'test'},
                'test', 'system', [{'function': {'name': 'range_search'}}], None, None, budget)
        return requests, record

    async def test_notices_preserve_budget(self):
        budget = 7
        requests, record = await self.run_loop(budget=budget)
        self.assertEqual(len(requests), budget)
        self.assertEqual(record['output']['metrics']['turn_count'], budget)
        notices = [json.dumps(request['messages'][-1]) for request in requests]
        self.assertIn('5 calls left', notices[-5])
        self.assertIn('2 calls left', notices[-2])
        self.assertTrue(any('repeated_tool_calls' in notice for notice in notices))
        self.assertEqual(requests[-1]['tool_choice'], 'none')
        self.assertEqual(len([e for e in record['trace']['events'] if e['type'] == 'tool_call']), budget - 1)
        calls = [event for event in record['trace']['events'] if event['type'] == 'llm_call']
        self.assertEqual([event['turn'] for event in calls], list(range(1, budget + 1)))

    async def test_saved_requests_reconstruct_exactly_without_shared_payload_duplication(self):
        requests, record = await self.run_loop(budget=3, base_url='https://openrouter.ai/api/v1')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'tasks.json'
            source.write_text('[{"task_id":"test","prompt":"test","answer":{"secret_truth":1}}]')
            args = SimpleNamespace(output=str(Path(directory) / 'run.json'), terminal_log=None,
                tasks=[str(source)], provider='openrouter', model='test', mcp_vendor='test',
                mcp_command='python', mcp_arg=['test.py'], mcp_env=['TOKEN=never-save-this'], max_agent_turns=3)
            with patch('builtins.print'), LogWriter(args, 'mcp', json.loads(source.read_text()), 'system',
                    [{'function': {'name': 'range_search'}}], {}) as log_writer:
                log_writer.record(0, record)
            document = json.loads(log_writer.path.read_text())
            task = document['tasks'][0]
            self.assertNotIn('secret_truth', json.dumps(document))
            self.assertNotIn('never-save-this', json.dumps(document))
            saved_calls = [event for event in task['trace']['events'] if event['type'] == 'llm_call']
            self.assertEqual(len(saved_calls), len(requests))
            for saved, original in zip(saved_calls, requests):
                request = saved['request']
                messages = copy.deepcopy(task['trace']['transcript'][:request['message_count']])
                messages[0]['content'] = document['system_prompt']['text']
                client = SimpleNamespace(base_url='https://openrouter.ai/api/v1')
                rebuilt = {'model': document['config']['model'],
                           'messages': api.cache_system(messages, client) if request.get('cache_system') else messages,
                           **document['config']['completion_kwargs'], **request['overrides']}
                if 'tools' in request:
                    rebuilt['tools'] = document['tools']['schemas']
                self.assertEqual(rebuilt, original)
            for event in task['trace']['events']:
                if event['type'] == 'tool_call':
                    self.assertEqual(task['trace']['transcript'][event['result_transcript_index']]['role'], 'tool')

    async def test_final_tools_are_not_executed(self):
        _, record = await self.run_loop(final='tools', budget=2)
        self.assertEqual(len([e for e in record['trace']['events'] if e['type'] == 'tool_call']), 1)
        self.assertIn('final answer-only call', record['output']['error'])

    async def test_final_answers_and_failures_are_reported_and_logged(self):
        replies = [None, response().model_copy(update={'choices': []}), response(''),
                   response('{', finish='length'), response('{"locations":[]}')]
        for module in (baseline, mcp):
            for reply in replies:
                valid = reply is replies[-1]
                with self.subTest(runner=module.__name__, reply=reply):
                    create = AsyncMock(return_value=reply)
                    client = SimpleNamespace(base_url='https://example.invalid',
                        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                    with patch.object(module, '_log'), patch.object(module, 'estimate_llm_cost', return_value=0):
                        if module is baseline:
                            record = await baseline.generate_baseline_answer(client, {'task_id': 'test'}, 'test', 'sys')
                        else:
                            record = await mcp.agent_loop(client, None, {'task_id': 'test'}, 'test', 'sys', [], None, None, 1)
                    create.assert_awaited_once()
                    self.assertEqual(record['output']['metrics']['zero_hit'], not valid)
                    self.assertEqual('error' in record['output'], not valid)
                    calls = [event for event in record['trace']['events'] if event['type'] == 'llm_call']
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]['response'], reply.model_dump(mode='json') if reply is not None else None)

    async def test_full_response_dump_preserves_metadata_and_original_text(self):
        reply = response('```json\n{"locations":[]}\n```')
        client = SimpleNamespace(base_url='https://example.invalid', close=AsyncMock(), chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=reply))))
        for module in [baseline, mcp]:
            kwargs = dict(task={'task_id': 'test'}, model='test', system_prompt='sys')
            with patch.object(module, '_log'), patch.object(module, 'estimate_llm_cost', return_value=0):
                if module is baseline:
                    record = await baseline.generate_baseline_answer(client, **kwargs)
                else:
                    record = await mcp.agent_loop(client, None, tools=[], known_tools=None,
                                                   knn_limit=None, max_agent_turns=2, **kwargs)
            self.assertEqual(record['output']['answer'], {'locations': []})
            dump = [e for e in record['trace']['events'] if e['type'] == 'llm_call'][0]['response']
            self.assertEqual(dump, reply.model_dump(mode='json'))
            json.dumps(record)

    async def test_json_validity_is_separate_from_execution_and_answer(self):
        for module in (baseline, mcp):
            for content, answer, valid in [
                ('{"answer": "hello"}', {'answer': 'hello'}, True),
                ('```json\n{"answer": "hello"}\n```', {'answer': 'hello'}, False),
                ('Sorry, service unavailable', None, False),
                ('{"answer":', None, False),
                ('null', None, True),
            ]:
                with self.subTest(runner=module.__name__, content=content):
                    reply = response(content)
                    client = SimpleNamespace(base_url='https://example.invalid', chat=SimpleNamespace(
                        completions=SimpleNamespace(create=AsyncMock(return_value=reply))))
                    with patch.object(module, '_log'), patch.object(module, 'estimate_llm_cost', return_value=0):
                        if module is baseline:
                            record = await baseline.generate_baseline_answer(client, {'task_id': 'test'}, 'test', 'sys')
                        else:
                            record = await mcp.agent_loop(client, None, {'task_id': 'test'}, 'test', 'sys', [], None, None, 1)
                    self.assertEqual(record['status'], 'completed')
                    self.assertNotIn('error', record['output'])
                    self.assertEqual(record['output']['answer'], answer)
                    self.assertEqual(record['output']['metrics']['json_valid'], valid)
                    self.assertEqual(record['output']['metrics']['zero_hit'], answer is None)
                    self.assertEqual(record['trace']['events'][0]['response'], reply.model_dump(mode='json'))

    async def test_custom_mcp_command_preserves_empty_arguments(self):
        with patch.object(sys, 'argv', ['runner', '--tasks', 'tasks.json', '--output', 'run.json',
                                       '--mcp-command', 'custom-server']):
            self.assertEqual(mcp.parse_args().mcp_arg, [])

    async def test_mcp_closes_session_on_setup_failure(self):
        client = SimpleNamespace(start=AsyncMock(), list_tools=AsyncMock(side_effect=RuntimeError('discovery failed')), close=AsyncMock())
        argv = ['runner', '--tasks', 'missing.json', '--output', 'run.json']
        with patch.object(sys, 'argv', argv), patch.object(mcp, 'build_openai_client', return_value=SimpleNamespace(close=AsyncMock())), \
             patch.object(mcp, 'AsyncMCPClient', return_value=client), patch.object(mcp, 'select_tasks', return_value=[]), \
             patch.object(client, 'close') as close, \
             patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'discovery failed'):
                await mcp.main()
            close.assert_called_once()

    async def test_mcp_tool_error_reaches_next_call(self):
        calls = []
        async def create(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return response(None, [Call()], 'tool_calls') if len(calls) == 1 else response('{}')
        client = SimpleNamespace(base_url='https://example.invalid', close=AsyncMock(), chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        tools = SimpleNamespace(call_tool=AsyncMock(side_effect=TimeoutError('mcp_tool_timeout_after_10s')))
        with patch.object(mcp, '_log'), patch.object(mcp, 'estimate_llm_cost', return_value=0):
            record = await mcp.agent_loop(client, tools, {'task_id': 'test'}, 'test', 'sys',
                                          [{'function': {'name': 'range_search'}}], None, None, 3)
        tool_message = next(m for m in calls[1]['messages'] if m['role'] == 'tool')
        self.assertIn('mcp_tool_timeout_after_10s', tool_message['content'])
        self.assertTrue([e for e in record['trace']['events'] if e['type'] == 'tool_call'][0]['timeout'])
        self.assertEqual(record['output']['metrics']['latency_breakdown_ms']['timeout_tool_count'], 1)

    async def test_tool_result_transport_is_compact_and_lossless(self):
        payload = {'count': 1, 'locations': [{'name': 'Café Main Street', 'lat': 40.4,
                   'tags': {'open': True, 'note': None}}]}
        requests = []
        async def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            return response(None, [Call()], 'tool_calls') if len(requests) == 1 else response('{}')
        client = SimpleNamespace(base_url='https://example.invalid', close=AsyncMock(), chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        tools = SimpleNamespace(call_tool=AsyncMock(return_value=payload))
        with patch.object(mcp, '_log'), patch.object(mcp, 'estimate_llm_cost', return_value=0):
            record = await mcp.agent_loop(client, tools, {'task_id': 'test'}, 'test', 'sys',
                [{'function': {'name': 'range_search'}}], None, None, 3)
        content = next(m['content'] for m in requests[1]['messages'] if m['role'] == 'tool')
        self.assertEqual(json.loads(content), payload)
        self.assertIn('Café Main Street', content)
        self.assertNotIn('"count": ', content)
        self.assertNotIn(', "locations"', content)
        self.assertLess(len(content), len(json.dumps(payload)))
        event = next(e for e in record['trace']['events'] if e['type'] == 'tool_call')
        self.assertEqual(json.loads(record['trace']['transcript'][event['result_transcript_index']]['content']), payload)

    async def test_both_main_paths_write_evaluator_files(self):
        for module in [baseline, mcp]:
            with self.subTest(runner=module.__name__), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'tasks.json'
                path.write_text(json.dumps([{'task_id': 'a'}, {'task_id': 'b'}]))
                argv = ['runner', '--tasks', str(path), '--output', str(Path(tmp) / 'run.json'), '--default-ip-location']
                if module is baseline:
                    argv += ['--num-workers', '2']
                client = SimpleNamespace(base_url='https://example.invalid', close=AsyncMock(), chat=SimpleNamespace(
                    completions=SimpleNamespace(create=AsyncMock(return_value=response('{}')))))
                tool_client = SimpleNamespace(start=AsyncMock(), list_tools=AsyncMock(return_value=[]), close=AsyncMock())
                with patch.object(sys, 'argv', argv), patch.object(module, 'build_openai_client', return_value=client), \
                     patch.object(module, '_log'), patch.object(module, 'estimate_llm_cost', return_value=0), \
                     patch.object(mcp, 'AsyncMCPClient', return_value=tool_client), patch('builtins.print'):
                    await module.main()
                document = json.loads((Path(tmp) / 'run.json').read_text(encoding='utf-8'))
                results = [t['output'] for t in document['tasks']]
                self.assertEqual([t['task_id'] for t in document['tasks']], ['a', 'b'])
                self.assertTrue(all('trace' in task for task in document['tasks']))
                self.assertTrue(all(r['answer'] == {} for r in results))
                self.assertEqual(document['config']['tasks'][0], str(path.resolve()))



class SharedTests(unittest.IsolatedAsyncioTestCase):
    def test_cache_marks_only_initial_system_prompt(self):
        messages = [{'role': 'system', 'content': 'Shared prompt'},
                    {'role': 'user', 'content': 'Question'}]
        messages += [{'role': 'system', 'content': f'Turn {i}'} for i in range(6)]
        original = copy.deepcopy(messages)
        client = SimpleNamespace(base_url='https://openrouter.ai/api/v1')
        sent = api.cache_system(messages, client)
        self.assertEqual(sent[0]['content'], [{'type': 'text', 'text': 'Shared prompt',
                                             'cache_control': {'type': 'ephemeral'}}])
        self.assertEqual(sent[1:], original[1:])
        self.assertEqual(messages, original)
        client.base_url = 'https://api.anthropic.com/v1'
        self.assertEqual(api.cache_system(messages, client), original)


    def test_unknown_price_is_not_free(self):
        empty = {'input_cost_per_token': 0, 'output_cost_per_token': 0}
        with patch.object(api.litellm, 'get_model_info', return_value=empty), \
             patch.object(api, '_openrouter_price', return_value=None):
            self.assertIsNone(api.estimate_llm_cost('unknown', 100, 50))
        with patch.object(api.litellm, 'get_model_info', return_value=empty), \
             patch.object(api, '_openrouter_price', return_value=empty):
            self.assertEqual(api.estimate_llm_cost('known-free', 100, 50), 0)

    async def test_missing_cost_stays_null_in_both_runners(self):
        reply = response('{}')
        reply.usage = reply.usage.model_copy(update={'cost': None})
        for module in (baseline, mcp):
            client = SimpleNamespace(base_url='https://example.invalid', chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=reply))))
            with patch.object(module, '_log'), patch.object(module, 'estimate_llm_cost', return_value=None):
                if module is baseline:
                    record = await baseline.generate_baseline_answer(client, {'task_id': 'test'}, 'test', 'sys')
                else:
                    record = await mcp.agent_loop(client, None, {'task_id': 'test'}, 'test', 'sys', [], None, None, 1)
            self.assertIsNone(record['output']['metrics']['cost_usd'])
            self.assertIn('"cost_usd": null', json.dumps(record))

    def test_nonstandard_json_constants_are_rejected(self):
        for constant in ('NaN', 'Infinity', '-Infinity'):
            for content in (constant, '{"distance_m": ' + constant + '}',
                            '```json\n{"distance_m": ' + constant + '}\n```',
                            'Answer: {"distance_m": ' + constant + '}'):
                with self.subTest(content=content):
                    self.assertEqual(api.parse_response_content(content), (None, False))
        self.assertEqual(api.parse_response_content('{"name": "NaN"}'), ({'name': 'NaN'}, True))

    async def test_provider_settings_and_effort(self):
        for provider in ['genai', 'openrouter', 'openai']:
            args = SimpleNamespace(temperature=0, seed=42, max_tokens=4096,
                                   provider=provider, reasoning='low')
            kwargs = api.build_completion_kwargs(args)
            self.assertEqual(api.reasoning_of(kwargs), 'low')
            if provider != 'openrouter':
                self.assertEqual(kwargs['reasoning_effort'], 'low')
                self.assertNotIn('extra_body', kwargs)
            elif provider == 'openrouter':
                self.assertEqual(kwargs['extra_body'], {'reasoning': {'effort': 'low'}, 'usage': {'include': True}})
            args.reasoning = None
            self.assertIsNone(api.reasoning_of(api.build_completion_kwargs(args)))

    async def test_sdk_owns_retries(self):
        import httpx
        from openai import AsyncOpenAI
        attempts = []
        def handle(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(429, headers={'retry-after-ms': '1'},
                                      json={'error': {'message': 'rate limit', 'type': 'rate_limit'}})
            return httpx.Response(200, json=response().model_dump(mode='json'))
        async with AsyncOpenAI(api_key='test', max_retries=1,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            events = []
            reply, error, latency = await api.request_completion(client, 'test', [{'role': 'user', 'content': 'hi'}], {}, events)
        self.assertIsNone(error)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['response'], reply.model_dump(mode='json'))
        async with AsyncOpenAI(api_key='test', max_retries=0, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(429,
                    json={'error': {'message': 'rate limit', 'type': 'rate_limit'}})))) as client:
            events = []
            reply, error, latency = await api.request_completion(client, 'test', [], {}, events)
        self.assertIsNone(reply)
        self.assertIn('429', error)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]['response'])
        with patch.object(api, 'AsyncOpenAI') as constructor, patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}):
            api.build_openai_client('openai', max_retries=4)
            self.assertEqual(constructor.call_args.kwargs['max_retries'], 4)

    async def test_sequential_delay_and_empty_runs(self):
        tasks = list(range(4))
        async def worker(index, task):
            return {'task_id': str(task)}
        for workers in [1, 3]:
            with patch.object(api.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
                results = await api.run_tasks(tasks, worker, num_workers=workers, inter_task_delay=1)
            self.assertEqual(sleep.call_count, 3 if workers == 1 else 0)
        self.assertEqual(await api.run_tasks([], worker), [])
        with self.assertRaises(ValueError):
            await api.run_tasks(tasks, worker, num_workers=0)

    async def test_concurrency_is_bounded_and_workers_cancel_on_failure(self):
        active = peak = 0
        async def worker(index, task):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                if index == 0 and workers > 1:
                    await first_completed.wait()
                await asyncio.sleep(.01)
                completion_order.append(index)
                first_completed.set()
                return {'task_id': index}
            finally:
                active -= 1
        for workers in (1, 3):
            peak = 0
            first_completed = asyncio.Event()
            completion_order = []
            results = await api.run_tasks(list(range(6)), worker, num_workers=workers)
            self.assertEqual(peak, workers)
            self.assertEqual(active, 0)
            self.assertEqual([r['task_id'] for r in results], list(range(6)))
            if workers > 1:
                self.assertLess(completion_order.index(1), completion_order.index(0))
        async def failing(index, task):
            if index == 0:
                await asyncio.sleep(.001)
                raise RuntimeError('stop')
            return await worker(index, task)
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            await api.run_tasks([0, 1, 2], failing, num_workers=3)
        self.assertEqual(active, 0)

    async def test_task_selection_rejects_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tasks.json'
            path.write_text(json.dumps({'tasks': [{'task_id': 'a', 'operation': 'knn'}, {'task_id': 'b', 'operation': 'join'}]}))
            args = SimpleNamespace(tasks=[str(path)], operation='join', task_ids=['b'], limit=1,
                                   output=str(Path(tmp) / 'run.json'), terminal_log=None)
            self.assertEqual(api.select_tasks(args), [{'task_id': 'b', 'operation': 'join'}])
            path.write_text('[{}]')
            args.operation = args.task_ids = args.limit = None
            with self.assertRaises(ValueError):
                api.select_tasks(args)


if __name__ == '__main__':
    unittest.main()
