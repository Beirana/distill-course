"""CPU-only correctness tests. These do not claim that AutoDL/CUDA has passed."""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common
import course
import verify_bundle


class AnswerTests(unittest.TestCase):
    def test_numeric_equivalence(self):
        for text in ("Answer: 42", "Answer: 42.", "Answer: 42.0", "Answer: +42", "Answer: 4.2e1"):
            with self.subTest(text=text):
                self.assertEqual(common.parse_answer(text)["value"], "42")
                self.assertTrue(common.parse_answer(text)["format_ok"])

    def test_sign_grouping_fraction_and_unicode(self):
        for text, expected in (("Answer: -3", "-3"), ("Answer: 1,234.50", "2469/2"),
                               ("Answer: 1/2", "1/2"), ("Answer: −3。", "-3")):
            self.assertEqual(common.parse_answer(text)["value"], expected)

    def test_does_not_score_an_intermediate_or_ambiguous_number(self):
        for text in ("We have 42 apples", "Answer: 42 or 43", "Answer: 12,34", "Answer: 1/0",
                     "Answer: 42\nBut I have not finished.", "Answer: NaN", "Answer: 42=41", ""):
            self.assertIsNone(common.parse_answer(text)["value"], text)

    def test_fallback_separates_math_from_format(self):
        for text in ("42", r"\boxed{42}", "#### 42"):
            self.assertEqual(common.parse_answer(text)["value"], "42")
            self.assertFalse(common.parse_answer(text)["format_ok"])

    def test_gold_requires_explicit_reference(self):
        self.assertEqual(common.gold_answer("calculation\n#### -3"), "-3")
        with self.assertRaises(ValueError): common.gold_answer("The result is 3")


class SelectionTests(unittest.TestCase):
    def fixture(self):
        train = [{"question": f"Compute {i} plus one.", "answer": f"reason\n#### {i+1}"} for i in range(12)]
        long = "One farmer has five red boxes with many blue balls and wants 17 balls today."
        train.append({"question": long, "answer": "reason\n#### 17"})
        test = [dict(train[0]), {"question": "A unique unseen question?", "answer": "ref\n#### 5"},
                {"question": long.replace("17", "18"), "answer": "ref\n#### 18"}]
        numina = [{"source": "gsm8k", "problem": r["question"]} for r in train]
        numina += [dict(numina[1]), {"source": "gsm8k", "problem": "No human reference for this."}]
        cfg = {"seed": 42, "max_problem_chars": 600, "candidate_count": 5, "dev_count": 2, "test_count": 2}
        return numina, train, test, cfg

    def test_disjoint_reproducible_reference_linked_split(self):
        fixture = self.fixture()
        candidates, dev, test, report = common.select_data(*fixture)
        again = common.select_data(*fixture)
        self.assertEqual((candidates, dev, test, report), again)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(dev), 2)
        self.assertFalse({r["id"] for r in candidates} & {r["id"] for r in dev})
        all_test = {common.normalize_question(r["question"]) for r in fixture[2]}
        self.assertFalse(all_test & {common.normalize_question(r["question"]) for r in candidates+dev})
        self.assertEqual(report["counts"]["test_exact_overlap_removed"], 1)
        self.assertEqual(report["counts"]["near_overlap_removed"], 1)
        self.assertEqual(report["counts"]["duplicate"], 1)
        self.assertEqual(report["counts"]["not_exactly_matched_to_human_train"], 1)
        self.assertTrue(all("reference_solution" in r for r in candidates))

    def test_fails_instead_of_pretending_to_have_enough_data(self):
        numina, train, test, cfg = self.fixture()
        cfg["candidate_count"] = 10000
        with self.assertRaisesRegex(ValueError, "Do not silently"):
            common.select_data(numina, train, test, cfg)

    def test_conflicting_gold_is_an_error(self):
        numina, train, test, cfg = self.fixture()
        train.append(dict(train[1], answer="#### 99"))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            common.select_data(numina, train, test, cfg)


class ArtifactTests(unittest.TestCase):
    def make_external_model(self, root, role, revision):
        folder = root / "models" / role
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "model.safetensors").write_bytes((role + " weights").encode())
        files = []
        for path in sorted(folder.iterdir()):
            files.append({"path": path.name, "url": f"https://example.invalid/{path.name}",
                          "size": path.stat().st_size, "sha256": common.digest_file(path)})
        manifest = root / (role + "-source.json")
        common.write_json(manifest, {
            "schema_version": "1.0",
            "source": {"provider": "test", "repo_id": course.CFG[role],
                       "requested_revision": "main", "revision": revision},
            "files": files,
        })
        return manifest

    def test_training_environment_can_use_a_validated_alternative(self):
        with patch.dict(course.os.environ, {}, clear=True):
            self.assertEqual(Path(course.training_env(Path('/data'))), Path(sys.prefix))
        with patch.dict(course.os.environ, {'COURSE_TRAIN_ENV': '/data/envs/validated-alternative'}):
            self.assertEqual(Path(course.training_env(Path('/data'))), Path('/data/envs/validated-alternative'))

    def test_no_overwrite_and_safe_run_names(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            common.write_json(path, {"first": True})
            with self.assertRaises(FileExistsError): common.write_json(path, {"first": False})
            self.assertEqual(common.read_json(path), {"first": True})
            for name in ("../outside", "/root", "a/b", ""):
                with self.assertRaises(RuntimeError): course.run_path(Path(d), name)

    def test_smoke_and_pilot_configs(self):
        root = Path("/root/distill-work")
        smoke, export = common.training_configs(root, root / "runs/smoke01", course.CFG, "smoke")
        pilot, _ = common.training_configs(root, root / "runs/pilot500-01", course.CFG, "pilot500")
        self.assertEqual(smoke["max_steps"], 2)
        self.assertNotIn("max_steps", pilot)
        self.assertEqual(pilot["num_train_epochs"], 3)
        self.assertEqual(pilot["cutoff_len"], 2048)
        self.assertFalse(pilot["train_on_prompt"])
        self.assertFalse(pilot["overwrite_output_dir"])
        self.assertEqual(export["export_device"], "cpu")

    def test_make_config_and_fork_preserve_data_but_not_trained_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "runs/original"
            common.write_json(folder / "dataset/train.json", [{"instruction": "x", "output": "Answer: 2"}])
            common.write_json(folder / "dataset/dataset_info.json", {})
            common.write_json(folder / "generation_complete.json", {"mode": "smoke", "train_sha256": common.digest_file(folder / "dataset/train.json")})
            common.write_json(folder / "request.json", {})
            common.write_jsonl(folder / "generation_audit.jsonl", [])
            course.make_config(root, argparse.Namespace(run="original"))
            course.fork_run(root, argparse.Namespace(run="second", source_run="original"))
            second = root / "runs/second"
            self.assertEqual(common.digest_file(folder / "dataset/train.json"), common.digest_file(second / "dataset/train.json"))
            self.assertNotEqual(common.read_json(folder / "train_config.yaml")["output_dir"], common.read_json(second / "train_config.yaml")["output_dir"])
            self.assertFalse((second / "adapter").exists())

    def test_fingerprint_detects_model_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"test fixture, not actual weights")
            files = course.inventory(model)
            course.verify_inventory(model, files)
            (model / "model.safetensors").write_bytes(b"changed")
            with self.assertRaises(RuntimeError): course.verify_inventory(model, files)

    def test_registers_verified_external_models_without_downloading(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            teacher = self.make_external_model(root, "teacher", "a" * 40)
            student = self.make_external_model(root, "student", "b" * 40)
            course.register_models(root, argparse.Namespace(
                teacher_source_manifest=str(teacher), student_source_manifest=str(student)))
            for role, revision in (("teacher", "a" * 40), ("student", "b" * 40)):
                registered = course.model_ready(root, role)
                self.assertEqual(registered["resolved_revision"], revision)
                self.assertTrue(registered["revision_is_immutable"])
                self.assertEqual(registered["registration"], "external-verified-manifest")

    def test_external_registration_is_all_or_nothing_on_hash_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            teacher = self.make_external_model(root, "teacher", "a" * 40)
            student = self.make_external_model(root, "student", "b" * 40)
            data = common.read_json(student)
            data["files"][0]["sha256"] = "0" * 64
            student.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs from external manifest"):
                course.register_models(root, argparse.Namespace(
                    teacher_source_manifest=str(teacher), student_source_manifest=str(student)))
            self.assertFalse((root / "models/teacher-manifest.json").exists())
            self.assertFalse((root / "models/student-manifest.json").exists())

    def test_missing_registration_explains_local_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_external_model(root, "teacher", "a" * 40)
            with self.assertRaisesRegex(RuntimeError, "registration is missing.*register-models"):
                course.model_ready(root, "teacher")

    def test_register_student_while_teacher_pending_and_keep_existing_registration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            student = self.make_external_model(root, "student", "b" * 40)
            args = argparse.Namespace(teacher_source_manifest=None, student_source_manifest=str(student))
            course.register_models(root, args)
            course.model_ready(root, "student")
            self.assertFalse((root / "models/teacher-manifest.json").exists())
            before = (root / "models/student-manifest.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                course.register_models(root, args)
            self.assertEqual(before, (root / "models/student-manifest.json").read_bytes())

    def test_registration_rejects_mutable_revision_and_wrong_role(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self.make_external_model(root, "student", "master")
            with self.assertRaisesRegex(RuntimeError, "immutable repository commit"):
                course.build_external_registration(root, "student", source)
            data = common.read_json(source)
            data["source"]["revision"] = "b" * 40
            source.write_text(json.dumps(data), encoding="utf-8")
            course.register_models(root, argparse.Namespace(teacher_source_manifest=None, student_source_manifest=str(source)))
            manifest = root / "models/student-manifest.json"
            data = common.read_json(manifest)
            data["model_id"] = "wrong/model"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                course.model_ready(root, "student")

    def test_bad_index_and_noncanonical_paths_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self.make_external_model(root, "teacher", "a" * 40)
            data = common.read_json(source)
            for path in ("./config.json", "a//b", "C:/escape", "line\nbreak"):
                changed = dict(data, files=[dict(data["files"][0], path=path)])
                source.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Unsafe external manifest path"):
                    course.external_model_manifest(source, "teacher")
            folder = root / "models/teacher"
            common.write_json(folder / "model.safetensors.index.json", {"weight_map": {"weight": []}})
            with self.assertRaisesRegex(RuntimeError, "unregistered shard"):
                course.verify_model_semantics(folder, course.inventory(folder))

    def test_external_manifest_rejects_unsafe_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "source.json"
            common.write_json(path, {
                "schema_version": "1.0",
                "source": {"provider": "test", "repo_id": course.CFG["teacher"], "revision": "a" * 40},
                "files": [{"path": "../escape", "size": 1, "sha256": "0" * 64}],
            })
            with self.assertRaisesRegex(RuntimeError, "Unsafe external manifest path"):
                course.external_model_manifest(path, "teacher")

    def test_report_uses_actual_denominator_and_marks_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common.write_json(root / "runs/test/eval_dev/before_summary.json", {
                "split": "dev", "model": "before", "n": 5, "correct": 2, "accuracy": .4,
                "format_rate": .6, "parse_rate": .8, "truncated_rate": .2,
                "batch_generate_seconds": 5, "output_tokens_per_second": 10, "weights_gib": .9})
            course.report(root, argparse.Namespace(run="test"))
            csv_path = next((root / "runs/test").glob("results-*.csv"))
            with csv_path.open(encoding="utf-8-sig") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["n"], "5")
            md = next((root / "runs/test").glob("results-*.md")).read_text(encoding="utf-8")
            self.assertIn("complete: False", md)

    def test_cli_help_needs_no_gpu(self):
        proc = subprocess.run([sys.executable, str(course.CODE / "scripts/course.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("prepare-data", proc.stdout)

    def test_freeze_rejects_config_changed_after_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "runs/pilot"
            common.write_json(folder / "generation_complete.json", {"mode": "pilot500"})
            common.write_json(folder / "train_config.yaml", {"learning_rate": .0001})
            common.write_json(folder / "export_config.yaml", {})
            common.write_json(folder / "train_complete.json", {"config_sha256": common.digest_file(folder / "train_config.yaml")})
            common.write_json(folder / "export_complete.json", {"config_sha256": common.digest_file(folder / "export_config.yaml")})
            (folder / "train_config.yaml").write_text('{"learning_rate": 0.1}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Training config changed"):
                course.freeze(root, argparse.Namespace(run="pilot"))
            self.assertFalse((folder / "frozen.json").exists())

    def test_eval_fingerprint_detects_data_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common.write_jsonl(root / "data/dev.jsonl", [{"question": "x", "gold": "1"}])
            before = course.evaluation_fingerprint(root, "dev")
            (root / "data/dev.jsonl").write_text('{"question": "x", "gold": "2"}\n', encoding="utf-8")
            self.assertNotEqual(before, course.evaluation_fingerprint(root, "dev"))

    def test_transfer_manifest_detects_missing_and_changed_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "lesson.md").write_text("correct", encoding="utf-8")
            common.write_json(root / "bundle-manifest.json", {"files": {"lesson.md": common.digest_file(root / "lesson.md")}})
            self.assertEqual(verify_bundle.verify(root), [])
            (root / "lesson.md").write_text("changed", encoding="utf-8")
            self.assertIn("Hash mismatch", verify_bundle.verify(root)[0])

    def test_transfer_manifest_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common.write_json(root / "bundle-manifest.json", {"files": {"../outside": "fakehash"}})
            self.assertIn("Unsafe", verify_bundle.verify(root)[0])

    def test_cli_local_data_preparation_reuse_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            source, root = Path(d) / "source", Path(d) / "runtime"
            train = [{"question": f"Compute {i} plus one.", "answer": f"reference\n#### {i+1}"} for i in range(1500)]
            tests = [{"question": f"Compute a separate subtraction {i}.", "answer": f"reference\n#### {i}"} for i in range(350)]
            numina = [{"source": "gsm8k", "problem": r["question"]} for r in train]
            for name, rows in (("numina", numina), ("gsm8k_train", train), ("gsm8k_test", tests)):
                common.write_jsonl(source / (name+".jsonl"), rows)
            cmd = [sys.executable, str(course.CODE / "scripts/course.py"), "--root", str(root),
                   "prepare-data", "--local-data", str(source)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(len(common.read_jsonl(root / "data/candidates.jsonl")), 1000)
            self.assertEqual(len(common.read_jsonl(root / "data/dev.jsonl")), 100)
            self.assertEqual(len(common.read_jsonl(root / "data/test.jsonl")), 300)
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            (root / "data/dev.jsonl").write_text('{"changed": true}\n', encoding="utf-8")
            third = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(third.returncode, 0)
            self.assertIn("Prepared data changed", third.stderr)
            statuses = [common.read_json(p)["status"] for p in sorted((root / "events").glob("*.json"))]
            self.assertEqual(statuses, ["complete", "complete", "failed"])


if __name__ == "__main__":
    unittest.main()
