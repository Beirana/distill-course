"""Linux supervisor for course.py generate/eval. Cleans only its own new process group.

Pass the same arguments as course.py. A forced cleanup is recorded and returns 2,
so a worker hang is never presented as a normally completed stage.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup(pgid):
    if not alive(pgid):
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + 5
    while alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def main():
    if os.name != 'posix':
        raise SystemExit('This process supervisor is for the AutoDL Linux host.')
    args = sys.argv[1:]
    if not any(x in args for x in ('generate', 'eval')):
        raise SystemExit('Use this wrapper only for course.py generate/eval.')
    root = Path(os.environ.get('COURSE_DATA_ROOT', '/root/distill-work'))
    if '--root' in args:
        root = Path(args[args.index('--root') + 1])
    logs = root / 'environment'
    logs.mkdir(parents=True, exist_ok=True)
    supervisor_dir = Path(tempfile.mkdtemp(prefix='process-', dir=logs))
    env = dict(os.environ, COURSE_STAGE_STATUS=str((supervisor_dir / 'stage.json').resolve()))
    proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('course.py')), *args],
                            env=env, start_new_session=True)
    completed_at = None
    stage = None
    forced = False
    try:
        while proc.poll() is None:
            status = supervisor_dir / 'stage.json'
            if status.exists():
                try:
                    stage = json.loads(status.read_text(encoding='utf-8'))
                    if completed_at is None:
                        completed_at = time.monotonic()
                except json.JSONDecodeError:
                    pass
            if completed_at is not None and time.monotonic() - completed_at > 20:
                forced = cleanup(proc.pid)
                break
            time.sleep(0.2)
        code = proc.wait(timeout=10)
    finally:
        forced = cleanup(proc.pid) or forced
    record = {'pid': proc.pid, 'process_group': proc.pid, 'returncode': code,
              'forced_cleanup': forced, 'stage_status': stage.get('status') if stage else None,
              'note': 'Only this supervisor-created process group was targeted. No name-based pkill.'}
    (supervisor_dir / 'supervisor.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    if forced:
        print('Outputs may be complete, but worker lifecycle needs review; not a clean workflow pass.')
        return 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
