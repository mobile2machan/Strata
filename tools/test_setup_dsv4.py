"""Tests for setup.py's DeepSeek-V4-Flash family (docs/DSV4.md): the pinned UD-IQ2_XXS (3 shards) and UD-IQ3_XXS
(4 shards) with their sizes and SHA-256, the per-size shard counts, the serve path's start args (--dsv4, no PLE
table, no draft layer, no KV flags), the pack built with --experts-bin, no images, one NVIDIA card, and the engine
version it needs.  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_dsv4
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REV = "fbbb5b93fb787c21338159b0af3318bb3f4d9768"
IQ2, IQ3 = "UD-IQ2_XXS", "UD-IQ3_XXS"


def names(model):
    fam = setup.FAMILIES["dsv4"]
    n = setup.shard_count(fam, model)
    return [fam["file"].format(q=model, i=i, n=n) for i in range(1, n + 1)]


class Pins(unittest.TestCase):
    def test_the_seven_shards(self):
        self.assertEqual(list(setup.DSV4_SHARDS)[:3], names(IQ2))
        self.assertEqual(list(setup.DSV4_SHARDS)[3:], names(IQ3))
        self.assertEqual(sum(b for k, (b, _) in setup.DSV4_SHARDS.items() if IQ2 in k), 90860736928)
        self.assertEqual(sum(b for k, (b, _) in setup.DSV4_SHARDS.items() if IQ3 in k), 104207848032)
        for _, sha in setup.DSV4_SHARDS.values():
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        fam = setup.FAMILIES["dsv4"]
        self.assertEqual(fam["hf"].format(q=IQ2),
                         f"https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/{REV}/UD-IQ2_XXS/")
        self.assertEqual(setup.MODELS[IQ2]["families"], ("dsv4",))       # no other family lists it
        self.assertTrue(fam["experimental"])
        self.assertEqual(list(setup.FAMILIES)[0], "qwen")                # not the default choice

    def test_shard_counts(self):
        fam = setup.FAMILIES["dsv4"]
        self.assertEqual(setup.shard_count(fam, IQ2), 3)
        self.assertEqual(setup.shard_count(fam, IQ3), 4)
        self.assertEqual(setup.shard_count(setup.FAMILIES["qwen"], "IQ3_XXS"), 2)
        self.assertEqual(setup.shard_count(setup.FAMILIES["unsloth"], "UD-Q4_K_XL"), 4)
        self.assertEqual(setup.shard_count(fam, "SOMETHING_ELSE"), fam.get("shards", 2))   # unknown: the family's

    def test_the_names_are_the_published_ones(self):
        for m in (IQ2, IQ3):
            for n in names(m):
                self.assertIsNone(setup.gguf_unsupported(n), n)
            self.assertEqual(setup.gguf_choice(names(m)[0]), ("dsv4", m))

    def test_a_qwen_named_ud_iq3_xxs_is_still_not_usable(self):
        # the #444 rule: the name says which model it is, and Qwen3.8-Flash-Next has no UD-IQ3_XXS
        self.assertEqual(setup.gguf_unsupported("Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"), IQ3)


class Config(unittest.TestCase):
    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "strata-dsv4-ud-iq2_xxs.json"
            p.write_text(json.dumps({"args": ["--max-context", "8192"]}))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"]), ("dsv4", IQ2))


def quiet(fn, *args):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Main(unittest.TestCase):
    """setup.main() for the DeepSeek-V4 choice, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, model=IQ2, ram=128.0, version="0.1.37", n_gpus=1, amd=(), dsv4=True, experts_ram=True):
        eng = self.t / "engine"
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({"version": version, "source": "local"}))
        found = [{"index": i, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "arch": "120", "driver": "580.97"}
                 for i in range(n_gpus)]
        self.downloads, self.runs, self.verified = [], [], []

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_run(cmd, *a, **k):
            self.runs.append([str(x) for x in cmd])

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: list(amd)),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "engine_has_arg", lambda exe, flag: {
                "--dsv4": dsv4, "--dsv4-experts-ram": experts_ram}.get(flag, False)),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.object(sys, "argv", ["setup.py", "--family", "dsv4", *(["--model", model] if model else []),
                                            "--yes", "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", mock.Mock(side_effect=AssertionError("asked"))),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = self.t / f"strata-dsv4-{(model or IQ2).lower()}.json"
        return code, out, (json.loads(cfg.read_text()) if cfg.exists() else None)

    def test_install(self):
        code, out, cfg = self.main(["--context", "8192"])
        self.assertEqual(code, 0, out)
        base = f"https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/{REV}/UD-IQ2_XXS/"
        self.assertEqual(self.downloads, [base + n for n in names(IQ2)])   # no image encoder
        self.assertEqual([v[0] for v in self.verified], names(IQ2))
        self.assertEqual([v[1:] for v in self.verified],
                         [setup.DSV4_SHARDS[n] for n in names(IQ2)])
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        self.assertIn("--experts-bin", packs[0])
        self.assertTrue(packs[0][packs[0].index("--gguf") + 1].endswith("UD-IQ2_XXS-00001-of-00003.gguf"))
        self.assertFalse(any(r[1].endswith(("mtp_fetch.py", "mtp_pack.py", "mtp_rt.py")) for r in self.runs))
        args = cfg["args"]
        self.assertEqual(args[:4], ["--pack", str(self.t / "data" / "packs" / "dsv4-ud-iq2_xxs"), "--dsv4",
                                    str(self.t / "models" / "dsv4-UD-IQ2_XXS" / names(IQ2)[0])])
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(args[args.index("--dsv4-experts-ram") + 1], "auto")   # this engine knows the option
        for flag in ("--native", "--ple-gguf", "--mtp", "--spec", "--expert-profile", "--expert-cache",
                     "--kv", "--kv-resident", "--rope-scaling", "--mmap-experts", "--resident-experts",
                     "--vision", "--control-vector-scaled"):
            self.assertNotIn(flag, args)
        self.assertEqual(cfg["model_name"], "deepseek-v4-flash-ud-iq2_xxs")
        self.assertNotIn("vision", cfg)
        self.assertNotIn("draft_vocab", cfg)
        self.assertNotIn("layer_split", cfg)

    def test_iq3_xxs_is_four_shards(self):
        code, out, cfg = self.main(["--context", "8192"], model=IQ3)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.downloads), 4)
        self.assertTrue(all(n.endswith(f"-of-00004.gguf") for n in self.downloads))

    def test_engine_without_the_dsv4_path_stops_before_the_download(self):
        # the published ready-made engine (even a newer version) may not carry --dsv4: setup asks the binary
        code, out, cfg = self.main(["--context", "8192"], dsv4=False)
        self.assertEqual(code, 1)
        self.assertIn("has no --dsv4 serve path", out)
        self.assertIn("--build", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)

    def test_engine_without_the_residency_option_omits_the_flag(self):
        # an engine from before --dsv4-experts-ram (the published ready-made one) is not handed the flag
        code, out, cfg = self.main(["--context", "8192"], experts_ram=False)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--dsv4-experts-ram", cfg["args"])

    def test_too_little_ram(self):
        code, out, cfg = self.main(["--context", "8192"], ram=95.9, model=None)   # --yes alone: still a stop
        self.assertEqual(code, 1)
        self.assertIn("needs about 110 GB of RAM", out)
        self.assertIn("no smaller size", out)                                     # this family has none
        self.assertIn("--model UD-IQ2_XXS --yes", out)
        self.assertEqual(self.downloads, [])

    def test_experimental_says_so(self):
        code, out, cfg = self.main(["--context", "8192"])
        self.assertEqual(code, 0, out)
        self.assertIn("EXPERIMENTAL (docs/DSV4.md)", out)
        self.assertIn("no drafter yet", out)

    def test_amd_refused(self):
        r9700 = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "arch": "gfx1201",
                  "driver": "amdgpu"}]
        code, out, cfg = self.main(["--context", "8192", "--backend", "hip"], amd=r9700)
        self.assertEqual(code, 1, out)
        self.assertIn("runs on NVIDIA cards only", out)
        self.assertEqual(self.downloads, [])

    def test_one_gpu_and_no_low_ram_mode(self):
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1", "--low-ram", "on"], n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertIn("runs on one card", out)
        self.assertIn("no low-RAM mode yet", out)
        self.assertNotIn("layer_split", cfg)
        for flag in ("--mmap-experts", "--resident-experts"):
            self.assertNotIn(flag, cfg["args"])

    def test_long_context_no_kv_flags(self):
        code, out, cfg = self.main(["--context", "131072"])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "131072")
        for flag in ("--kv", "--kv-resident"):
            self.assertNotIn(flag, args)


if __name__ == "__main__":
    unittest.main()
