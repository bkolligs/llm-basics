"""Print top-level runtime pins from uv.lock; does not install or modify anything."""
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parents[1]
lock = tomllib.loads((root / 'uv.lock').read_text())
project = tomllib.loads((root / 'pyproject.toml').read_text())
for requirement in project['project']['dependencies']:
    name = requirement.split('>')[0].split('<')[0].split('=')[0]
    versions = sorted({package['version'] for package in lock['package'] if package['name'] == name})
    if len(versions) != 1:
        raise SystemExit(f'Expected one locked version for {name}, got {versions}')
    print(f'{name}=={versions[0]}')
