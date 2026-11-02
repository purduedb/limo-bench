"""Session fixtures `oracle` / `fx` and a per-tool-call latency report at session end (harness.LATENCY_LOG)."""
import pytest

import harness
from fixtures import Fixtures
from oracle import Oracle


@pytest.fixture(scope="session")
def oracle():
    return Oracle()


@pytest.fixture(scope="session")
def fx(oracle):
    return Fixtures(oracle)


def pytest_runtest_setup(item):
    harness._current_test_id = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    log = harness.LATENCY_LOG
    if not log:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    reporter.write_sep("=", "per-tool latency (by tool)")
    by_tool: dict[str, list[float]] = {}
    for _, tool, dt in log:
        by_tool.setdefault(tool, []).append(dt)
    rows = sorted(by_tool.items(), key=lambda kv: -sum(kv[1]))
    reporter.write_line(f"{'tool':<24}{'calls':>7}{'total_s':>10}{'avg_s':>9}{'max_s':>9}")
    for tool, times in rows:
        reporter.write_line(
            f"{tool:<24}{len(times):>7}{sum(times):>10.3f}{sum(times) / len(times):>9.3f}{max(times):>9.3f}"
        )

    reporter.write_sep("=", "per-test latency (top 20 slowest, tool-call time only)")
    by_test: dict[str, list[float]] = {}
    for test_id, _, dt in log:
        by_test.setdefault(test_id, []).append(dt)
    test_rows = sorted(by_test.items(), key=lambda kv: -sum(kv[1]))[:20]
    reporter.write_line(f"{'total_s':>9}{'calls':>7}  test")
    for test_id, times in test_rows:
        short_test = test_id.split("::")[-1]
        reporter.write_line(f"{sum(times):>9.3f}{len(times):>7}  {short_test}")

    reporter.write_sep("=", "per-tool latency (slowest individual calls)")
    for test_id, tool, dt in sorted(log, key=lambda r: -r[2])[:15]:
        short_test = test_id.split("::")[-1]
        reporter.write_line(f"{dt:>8.3f}s  {tool:<24}{short_test}")
