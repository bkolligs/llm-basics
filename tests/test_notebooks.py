"""Infrastructure checks only: no answers to learner exercises live here."""
import ast
import io
import json
from pathlib import Path
import re
import runpy
from types import SimpleNamespace
import time

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / 'notebooks').glob('*.py'))


def load_definitions(path, device='cpu'):
    """Load pure function definitions and check lists without executing the UI."""
    namespace = dict(torch=torch, np=np, plt=plt, time=time, device=torch.device(device))
    tree = ast.parse(path.read_text())
    for cell in tree.body:
        if not isinstance(cell, ast.FunctionDef):
            continue
        nodes = [node for node in cell.body if isinstance(node, ast.FunctionDef)]
        nodes += [node for node in cell.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'exercise_cases' for t in node.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def cell_defining(path, name):
    for cell in ast.parse(path.read_text()).body:
        if isinstance(cell, ast.FunctionDef) and any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
            for node in ast.walk(cell)
        ):
            return cell
    raise AssertionError(f'Missing cell for {name}')


def execute_cell(cell, namespace):
    body = [node for node in cell.body if not isinstance(node, ast.Return)]
    exec(compile(ast.Module(body=body, type_ignores=[]), '<cell>', 'exec'), namespace)


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem)
def test_clean_open_never_trains(path):
    _, definitions = runpy.run_path(str(path))['app'].run()
    assert 'figure' not in definitions
    assert 'measurements' not in definitions
    assert all(row['status'] in {'TODO', 'FAIL', 'PASS'} for row in definitions['check_results'])
    # Once a learner completes the lab, opening it must still not train.
    if definitions['checks_passed']:
        assert not definitions['run_experiment'].value
    else:
        assert 'run_experiment' not in definitions


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem)
def test_device_selection_and_explicit_errors(path):
    resolve = load_definitions(path)['resolve_device']
    def backend(cuda, mps):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)), device=torch.device)
    for cuda, mps, expected in [(True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')]:
        assert resolve('auto', backend(cuda, mps)).type == expected
        assert resolve('cpu', backend(cuda, mps)).type == 'cpu'
    for name in ['cuda', 'mps']:
        with pytest.raises(RuntimeError, match='unavailable'):
            resolve(name, backend(False, False))
    with pytest.raises(ValueError, match='Unknown device'):
        resolve('tpu', backend(False, False))


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem)
def test_checks_separate_todo_failure_and_success(path):
    suite = load_definitions(path)['check_suite']
    def unfinished(): raise NotImplementedError('exercise')
    def wrong(): raise AssertionError('incorrect behavior')
    results = suite([('unfinished', unfinished), ('wrong', wrong), ('correct', lambda: None)])
    assert [row['status'] for row in results] == ['TODO', 'FAIL', 'PASS']
    assert 'incorrect behavior' in results[1]['detail']


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem)
def test_device_change_recreates_unpressed_run_control(path):
    cell = cell_defining(path, 'run_experiment')
    assert {'device', 'checks_passed'} <= {arg.arg for arg in cell.args.args}
    namespace = {'mo': mo, 'checks_passed': True, 'device': torch.device('cpu')}
    execute_cell(cell, namespace)
    old_button = namespace['run_experiment']
    # Simulate the frontend click; re-executing the device-dependent cell
    # must create a different unpressed control rather than retain True.
    old_button._update(1)
    assert old_button.value
    namespace['device'] = torch.device('cuda')  # no allocation required
    execute_cell(cell, namespace)
    assert namespace['run_experiment'] is not old_button
    assert namespace['run_experiment'].value is False
    with pytest.raises(mo.MarimoStopError):
        execute_cell(cell, dict(namespace, checks_passed=False))


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem)
def test_export_produces_json_and_png_without_disk_side_effects(path):
    figure, _ = plt.subplots(1, 3)
    namespace = dict(mo=mo, json=json, io=io, figure=figure, measurements=[{'value': 1.0}],
        failure_gallery=[{'case': 'fixture'}], device=torch.device('cpu'), torch=torch,
        check_results=[{'status': 'PASS'}])
    execute_cell(cell_defining(path, 'result_payload'), namespace)
    assert json.loads(json.dumps(namespace['result_payload']))['device'] == 'cpu'
    assert namespace['plot_buffer'].getvalue().startswith(b'\x89PNG')
    plt.close(figure)


def test_curriculum_links_and_durable_answers():
    readme = (ROOT / 'README.md').read_text()
    rows = re.findall(r'^\| (\d\d) \|', readme, re.M)
    assert rows == [f'{i:02}' for i in range(1, 35)]
    for target in re.findall(r'\]\(([^)]+)\)', readme):
        if '://' not in target and not target.startswith('#'):
            assert (ROOT / target).exists(), target
    for path in NOTEBOOKS:
        source = path.read_text()
        assert '[write here]' in source and 'Teach it back' in source
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                       ('llm_basics' in ast.unparse(node) or 'notebooks.' in ast.unparse(node))
                       for node in ast.walk(ast.parse(source)))


def test_timer_synchronizes_before_and_after():
    ns = load_definitions(NOTEBOOKS[0])
    events = []
    ns['synchronize'] = lambda device: events.append('sync')
    result, elapsed = ns['timed'](torch.device('cpu'), lambda: events.append('work'))
    assert events == ['sync', 'work', 'sync']
    assert result is None and elapsed >= 0
