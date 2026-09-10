"""Plan/install wheels in the CURRENT image environment. Never creates Python/Conda environments."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['gen', 'train'])
    p.add_argument('--apply', action='store_true', help='Install after checking the freshly resolved plan')
    p.add_argument('--requirements', type=Path, help='Alternative reviewed wheel profile')
    p.add_argument('--allow-torch-change', action='store_true', help='Explicit opt-in; record rationale before using')
    p.add_argument('--lf-ref', default='v0.9.5')
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('Image Torch cannot access CUDA: diagnose before installing anything.')
    root = Path(os.environ.get('COURSE_DATA_ROOT', '/root/distill-work'))
    env_dir = root / 'environment'
    env_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    prefix = env_dir / (args.stage + '-' + stamp)
    before = {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'torch': torch.__version__, 'torch_path': torch.__file__, 'cuda': torch.version.cuda}
    prefix.with_suffix('.before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    print(json.dumps(before, indent=2))
    if sys.version_info[:2] != (3, 12):
        print('NOTICE: image preference is Python 3.12; current version differs. No Python replacement is performed.')
    pip = [sys.executable, '-m', 'pip']
    freeze = run(pip + ['freeze'], capture_output=True, text=True).stdout
    prefix.with_suffix('.before-freeze.txt').write_text(freeze, encoding='utf-8')
    req = (args.requirements or CODE / 'configs' / ('requirements-' + args.stage + '.txt')).resolve()
    install = pip + ['install', '--only-binary=:all:', '-r', str(req)]
    if not args.allow_torch_change:
        constraint = prefix.with_suffix('.torch-constraint.txt')
        constraint.write_text('torch==' + importlib.metadata.version('torch') + '\n', encoding='utf-8')
        install += ['-c', str(constraint)]
    if args.stage == 'train':
        # LLaMA-Factory itself is local Python packaging, not a vLLM/CUDA source build.
        vendor = CODE / 'vendor' / 'LLaMA-Factory'
        if not vendor.exists():
            vendor.parent.mkdir(parents=True, exist_ok=True)
            run(['git', 'clone', '--depth', '1', '--branch', args.lf_ref,
                 'https://github.com/hiyouga/LLaMA-Factory.git', vendor])
        commit = run(['git', '-C', vendor, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        prefix.with_suffix('.lf-commit.txt').write_text(commit + '\n', encoding='utf-8')
        if args.lf_ref == 'v0.9.5' and commit != '7af909522a951e3ad9f022ea6f88b6755257eaa5':
            raise SystemExit('LLaMA-Factory checkout differs from the recorded v0.9.5 commit; inspect before continuing.')
        install += ['-e', str(vendor)]
    plan_path = prefix.with_suffix('.plan.json')
    run(install + ['--dry-run', '--report', str(plan_path)])
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    changes = [(i['metadata']['name'], i['metadata']['version']) for i in plan.get('install', [])]
    print('Planned changes:', changes)
    if not args.allow_torch_change and any(n.lower() == 'torch' for n, _ in changes):
        raise SystemExit('Resolver would reinstall Torch. Inspect the plan or choose compatible wheels; no install performed.')
    if not args.apply:
        print('PLAN ONLY: no pip installation performed. Review report, then rerun with --apply.')
        return
    run(install)
    run(pip + ['check'])
    # Fresh interpreter is essential: this process imported Torch before pip ran.
    check = 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__,torch.version.cuda)'
    check += '; import vllm; print(vllm.__version__)' if args.stage == 'gen' else '; import llamafactory; print("LLaMA-Factory import OK")'
    run([sys.executable, '-c', check])
    after = run(pip + ['freeze'], capture_output=True, text=True).stdout
    prefix.with_suffix('.after-freeze.txt').write_text(after, encoding='utf-8')
    print('Installed into', sys.prefix, '; GPU generation/training smoke checks still required.')


if __name__ == '__main__':
    main()
