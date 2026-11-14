from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from finetune.prepare_dataset import _publish, fade_edges, prepare, segment
from finetune.run_finetune import audio_files, run_dir_has_artifacts
from finetune.upstream_patch import ensure_full_state_resume


class PrepareDatasetTests(unittest.TestCase):
    def test_orphaned_backup_is_recovered_before_refusing_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "prepared"
            backup = root / ".prepared.backup-interrupted"
            backup.mkdir()
            (backup / "old.wav").write_bytes(b"old complete data")
            staging = root / ".prepared.staging-current"
            staging.mkdir()
            (staging / "new.wav").write_bytes(b"new data")
            with self.assertRaises(OSError):
                _publish(staging, output, overwrite=False)
            self.assertEqual((output / "old.wav").read_bytes(), b"old complete data")
            self.assertTrue((staging / "new.wav").is_file())

    @staticmethod
    def _write_tone(path: Path, frequency: float, seconds: float = 2.25, sr: int = 8_000) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(round(seconds * sr), dtype=np.float64) / sr
        pcm = (np.sin(2 * math.pi * frequency * t) * 9_000).astype("<i2")
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sr)
            out.writeframes(pcm.tobytes())

    def test_unique_paths_hashes_manifest_and_durations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "raw"
            self._write_tone(source / "speaker_a" / "same.wav", 220)
            self._write_tone(source / "speaker_b" / "same.wav", 330)
            out = base / "prepared"
            summary = prepare(source, out, sample_rate=8_000, min_sec=1.0, max_sec=2.0)
            rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            names = [row["path"] for row in rows]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(summary["segments"], len(rows))
            self.assertEqual(
                {row["source"] for row in rows}, {"speaker_a/same.wav", "speaker_b/same.wav"}
            )
            self.assertNotEqual(
                next(row["source_sha256"] for row in rows if row["source"].startswith("speaker_a")),
                next(row["source_sha256"] for row in rows if row["source"].startswith("speaker_b")),
            )
            for row in rows:
                segment_path = out / row["path"]
                self.assertTrue(segment_path.is_file())
                self.assertEqual(
                    hashlib.sha256(segment_path.read_bytes()).hexdigest(), row["segment_sha256"]
                )
                self.assertGreaterEqual(row["duration_s"], 1.0)
                self.assertLessEqual(row["duration_s"], 2.0 + 1e-6)
                self.assertEqual(row["sample_rate"], 8_000)
                self.assertEqual(row["channels"], 1)
                self.assertEqual(row["fade_ms"], 10)
                self.assertEqual(row["peak_target_dbfs"], -3.0)

    def test_existing_output_is_not_mixed_or_replaced_without_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "raw"
            self._write_tone(source / "x.wav", 220)
            out = base / "prepared"
            out.mkdir()
            marker = out / "user-data.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                prepare(source, out, sample_rate=8_000)
            self.assertEqual(marker.read_text(), "keep")
            prepare(source, out, sample_rate=8_000, overwrite=True)
            self.assertFalse(marker.exists())
            self.assertTrue((out / "manifest.jsonl").is_file())

    def test_invalid_audio_does_not_publish_empty_training_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "raw"
            source.mkdir()
            (source / "broken.ogg").write_bytes(b"not audio")
            out = base / "prepared"
            with self.assertRaises(RuntimeError):
                prepare(source, out, sample_rate=8_000)
            self.assertFalse(out.exists())

    def test_abandoned_staging_audio_is_not_treated_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "raw"
            self._write_tone(source / "good.wav", 220, seconds=1.5)
            stale = source / ".prepared.staging-abandoned"
            stale.mkdir()
            (stale / "partial.wav").write_bytes(b"not audio")
            summary = prepare(
                source, source / "prepared", sample_rate=8_000, min_sec=1.0, max_sec=2.0
            )
            self.assertEqual(summary["source_files"], 1)

    def test_symbolic_output_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "raw"
            self._write_tone(source / "good.wav", 220)
            actual = base / "actual-output"
            actual.mkdir()
            link = base / "prepared-link"
            try:
                link.symlink_to(actual, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable on this filesystem")
            with self.assertRaises(ValueError):
                prepare(source, link, sample_rate=8_000)

    def test_invalid_duration_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "raw"
            source.mkdir()
            with self.assertRaises(ValueError):
                prepare(source, base / "out", sample_rate=8_000, min_sec=2.0, max_sec=2.0)

    def test_segment_conserves_audio_and_respects_limits_when_possible(self) -> None:
        sr = 8_000
        x = np.sin(np.arange(int(5.25 * sr), dtype=np.float32) * 0.1)
        parts = segment(x, sr, max_sec=2.0, min_sec=1.0)
        self.assertTrue(parts)
        self.assertEqual(sum(part.size for part in parts), x.size)
        self.assertTrue(all(int(sr) <= part.size <= 2 * sr for part in parts))

    def test_fade_edges_removes_discontinuity_without_mutating_input(self) -> None:
        x = np.ones(1_000, dtype=np.float32)
        faded = fade_edges(x, 8_000, fade_ms=10)
        self.assertEqual(x[0], 1.0)
        self.assertEqual(faded[0], 0.0)
        self.assertEqual(faded[-1], 0.0)
        self.assertEqual(faded[80], 1.0)


class UpstreamResumePatchTests(unittest.TestCase):
    def test_audio_preflight_matches_upstream_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio.wav").write_bytes(b"wav")
            (root / "audio.AAC").write_bytes(b"aac")
            (root / "audio.WAV").write_bytes(b"upper suffix")
            self.assertEqual([p.name for p in audio_files(root)], ["audio.wav"])
            self.assertEqual(
                [p.name for p in audio_files(root, for_prepare=True)],
                ["audio.AAC", "audio.WAV", "audio.wav"],
            )

    def test_run_artifact_guard_distinguishes_fresh_empty_and_used_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing"
            empty = root / "empty"
            used = root / "used"
            empty.mkdir()
            used.mkdir()
            (used / "train_console.log").write_text("partial job")
            self.assertFalse(run_dir_has_artifacts(missing))
            self.assertFalse(run_dir_has_artifacts(empty))
            self.assertTrue(run_dir_has_artifacts(used))

    def test_patch_is_idempotent_and_only_changes_resume_mode(self) -> None:
        from finetune.upstream_patch import _OLD_LOAD, _OLD_TRAIN

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "import os\n" + _OLD_LOAD + "\n" + _OLD_TRAIN + "\n"
            (root / "train.py").write_text(source)
            self.assertTrue(ensure_full_state_resume(root))
            patched = (root / "train.py").read_text()
            self.assertIn("load_only_params=not is_training_resume", patched)
            self.assertIn("pretrained_ckpt_path is None", patched)
            self.assertIn("if self.iters < self.max_steps:", patched)
            self.assertIn("range(self.epoch, self.n_epochs)", patched)
            self.assertFalse(ensure_full_state_resume(root))
            self.assertEqual((root / "train.py").read_text(), patched)

    def test_unknown_upstream_layout_is_not_patched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "# unrelated train.py\n"
            (root / "train.py").write_text(source)
            with self.assertRaises(RuntimeError):
                ensure_full_state_resume(root)
            self.assertEqual((root / "train.py").read_text(), source)


if __name__ == "__main__":
    unittest.main()
