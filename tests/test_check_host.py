"""Exercise the host-check decisions with fake capabilities: no GPU or network calls."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = str(Path('C:/Program Files/Git/bin/bash.exe')) if os.name == 'nt' else shutil.which('bash')

HARNESS = r'''
python() {
  case "$1" in
    *verify_bundle.py) [ "$BAD_BUNDLE" = 0 ] || { echo 'fixture hash mismatch'; return 1; } ;;
    -c) case "$2" in *sys.prefix*) echo /fixture/runtime ;; *) echo 3.11 ;; esac ;;
    -) if [ "$BAD_GPU" = 1 ]; then echo 'fixture|12.8|False'; else echo 'fixture|12.8|True|fixture-gpu|32GiB|gpu_tensor_ok'; fi ;;
    -m) if [ "$3" = check ]; then echo 'unrelated web package dependency warning'; return 1; else echo 'Version: fixture'; fi ;;
    *) echo 'UNEXPECTED_PYTHON_CALL'; return 1 ;;
  esac
}
pip() { echo 'WRONG_PIP_INTERPRETER'; return 1; }
nvidia-smi() { case "$*" in *query-compute-apps*) : ;; *) echo 0 ;; esac; }
free() { echo 'Mem: 100 1 1 1 1 80'; }
df() { case "$*" in *--output=target*) printf 'Mounted on\n/fixture-disk\n' ;; *) printf 'Avail\n50G\n' ;; esac; }
timeout() { printf '000'; }
source "$CHECK_SCRIPT"
'''


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'A local Bash is needed for shell behavior tests')
class HostCheckTests(unittest.TestCase):
    def run_fixture(self, bad_gpu=False, bad_bundle=False):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ('runs', 'data'):
                (folder / name).mkdir()
            env = dict(os.environ, PYTHON='python', COURSE_DATA_ROOT=folder.as_posix(),
                       DATA_DISK_DIR=folder.as_posix(), COURSE_TRAIN_ENV=(folder/'no-training-env').as_posix(),
                       CHECK_SCRIPT=(ROOT/'scripts/check_host.sh').as_posix(),
                       BAD_GPU=str(int(bad_gpu)), BAD_BUNDLE=str(int(bad_bundle)))
            return subprocess.run([BASH, '-c', HARNESS], env=env, capture_output=True,
                                  text=True, encoding='utf-8', timeout=30)

    def test_offline_version_layout_and_unrelated_pip_warnings_do_not_block(self):
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('FAIL=0', result.stdout)
        self.assertIn('[WARN]', result.stdout)
        self.assertNotIn('WRONG_PIP_INTERPRETER', result.stdout)
        self.assertNotIn('UNEXPECTED_PYTHON_CALL', result.stdout)

    def test_missing_cuda_still_blocks_gpu_readiness(self):
        result = self.run_fixture(bad_gpu=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('[FAIL]', result.stdout)

    def test_unknown_bundle_is_not_silently_accepted(self):
        result = self.run_fixture(bad_bundle=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('fixture hash mismatch', result.stdout)


if __name__ == '__main__':
    unittest.main()
