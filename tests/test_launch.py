"""Exercise launch configuration without importing torch or allocating an NPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def run_launcher(self, script="launch.sh", **overrides):
        with tempfile.TemporaryDirectory(prefix="dual batch test ") as directory:
            folder = Path(directory)
            env_file = folder / "vendor-env.sh"
            env_file.write_text(":\n")
            interpreter = folder / "fake-python"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json,os,sys\n"
                "print('ARGV='+json.dumps(sys.argv[1:]))\n"
                "print('DUAL='+os.environ['DSV32_DUAL_BATCH'])\n"
                "print('OFFLINE='+os.environ['HF_HUB_OFFLINE'])\n"
            )
            interpreter.chmod(0o755)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("DSV32_")}
            env.update(CANN_ENV_FILE=str(env_file), ATB_ENV_FILE=str(env_file),
                       PYTHON=str(interpreter), DSV32_OUTPUT_DIR=str(folder / "output"),
                       HOST="127.0.0.1", PORT="18081")
            env.update(overrides)
            result = subprocess.run(["bash", str(ROOT / script)], cwd=directory,
                                    env=env, text=True, capture_output=True)
            return result

    def argv(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(next(line[5:] for line in result.stdout.splitlines()
                               if line.startswith("ARGV=")))

    def test_graph_and_relocated_paths(self):
        result = self.run_launcher()
        args = self.argv(result)
        self.assertEqual(args[args.index("--model") + 1], str(ROOT / "model"))
        self.assertEqual(args[args.index("--port") + 1], "18081")
        self.assertEqual(args[args.index("--load-format") + 1], "dummy")
        config = json.loads(args[args.index("--compilation-config") + 1])
        self.assertEqual(config["cudagraph_capture_sizes"], list(range(1, 9)))
        self.assertIn("DUAL=1", result.stdout)
        self.assertIn("OFFLINE=1", result.stdout)

    def test_eager_validation(self):
        result = self.run_launcher("launch.eager.sh", DSV32_DUAL_BATCH_VALIDATE="1")
        args = self.argv(result)
        self.assertIn("--enforce-eager", args)
        self.assertNotIn("--compilation-config", args)

    def test_baseline(self):
        result = self.run_launcher(DSV32_DUAL_BATCH="0")
        self.argv(result)
        self.assertIn("DUAL=0", result.stdout)

    def test_reject_graph_validation(self):
        result = self.run_launcher(DSV32_DUAL_BATCH_VALIDATE="1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires launch.eager.sh", result.stderr)

    def test_reject_unknown_mode(self):
        result = self.run_launcher(DSV32_EXECUTION_MODE="invalid")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
