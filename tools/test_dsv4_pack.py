"""deepseek4 pack tests (docs/DSV4.md P1); run: python -m unittest discover -s tools -p test_dsv4_pack.py

Builds a tiny two-shard deepseek4 GGUF in the artifact's own shape - metadata-only shard 1, tensors in
shard 2, a hash layer (no router, I32 tid2eid table) beside a routed layer, IQ2_XXS gate/up with IQ3_XXS
down - packs it with iq_pack, and checks what the pack must guarantee: the expert arena is the source
bytes relaid per expert, floats land in dense.bin unchanged, quantized projections and the hash table are
served natively, and model.json carries the geometry.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

from _paths import add_gguf_py
add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
import iq_pack

NE, H, FF, VOCAB = 4, 256, 256, 512          # H, FF multiples of 256: IQ blocks run along GGUF dim 0
GU_E = H * FF * 66 // 256                    # one IQ2_XXS gate/up expert
D_E = H * FF * 98 // 256                     # one IQ3_XXS down expert
BLOB = 2 * GU_E + D_E


def _opaque(w, name, gguf_shape, qtype, seed):
    """A quantized tensor the packer must treat as opaque bytes (gguf-py cannot encode IQ2_XXS/Q4_K).
    gguf-py takes shapes in numpy order and reverses them on write, and for uint8 data the last axis is
    BYTES, converted through the block geometry - so the byte shape here is reversed(gguf_shape) with the
    element axis scaled to bytes."""
    from gguf.constants import GGML_QUANT_SIZES
    ne, nb = GGML_QUANT_SIZES[qtype]
    np_shape = list(reversed(gguf_shape))
    assert np_shape[-1] % ne == 0                  # blocks run along GGUF dim 0 = numpy's last axis
    np_shape[-1] = np_shape[-1] * nb // ne
    data = np.random.default_rng(seed).integers(0, 256, int(np.prod(np_shape)), dtype=np.uint8)
    w.add_tensor(name, data.reshape(np_shape), raw_shape=np_shape, raw_dtype=qtype)


def _write_fixture(root):
    s1, s2 = root / "dsv4-fake-00001-of-00002.gguf", root / "dsv4-fake-00002-of-00002.gguf"
    n_tensors = 2 + 3 * 2 + 1 + 2 + 2 * 2 + 1 + 1          # embd/out, experts, tid2eid, router+bias, norms, kv, out_norm
    w1 = GGUFWriter(str(s1), "deepseek4")
    w1.add_uint16("split.no", 0)
    w1.add_uint16("split.count", 2)
    w1.add_int32("split.tensors.count", n_tensors)
    w1.add_uint32("deepseek4.block_count", 2)
    w1.add_uint32("deepseek4.embedding_length", H)
    w1.add_uint32("deepseek4.expert_count", NE)
    w1.add_uint32("deepseek4.expert_used_count", 2)
    w1.add_uint32("deepseek4.hash_layer_count", 1)
    w1.add_array("deepseek4.attention.compress_ratios", [0, 4])
    w1.write_header_to_file()
    w1.write_kv_data_to_file()
    w1.write_tensors_to_file()
    w1.close()

    rng = np.random.default_rng(11)
    w2 = GGUFWriter(str(s2), "deepseek4")
    w2.add_uint16("split.no", 1)
    w2.add_uint16("split.count", 2)
    w2.add_int32("split.tensors.count", n_tensors)
    _opaque(w2, "token_embd.weight", [H, VOCAB], Q.Q4_K, 1)
    _opaque(w2, "output.weight", [H, VOCAB], Q.Q4_K, 2)
    for l in (0, 1):
        _opaque(w2, "blk.%d.ffn_gate_exps.weight" % l, [H, FF, NE], Q.IQ2_XXS, 10 + l)
        _opaque(w2, "blk.%d.ffn_up_exps.weight" % l, [H, FF, NE], Q.IQ2_XXS, 20 + l)
        _opaque(w2, "blk.%d.ffn_down_exps.weight" % l, [FF, H, NE], Q.IQ3_XXS, 30 + l)
    # the hash layer: a static token -> expert table, and no router (GGUF [NE, VOCAB] = numpy (VOCAB, NE))
    tid = rng.integers(0, NE, (VOCAB, NE)).astype(np.int32)
    w2.add_tensor("blk.0.ffn_gate_tid2eid.weight", tid)
    # the routed layer
    router = rng.standard_normal((H, NE)).astype(np.float32)
    rb = quants.quantize(router, Q.BF16)
    w2.add_tensor("blk.1.ffn_gate_inp.weight", rb.reshape(NE, H * 2), raw_shape=[NE, H * 2], raw_dtype=Q.BF16)
    w2.add_tensor("blk.1.exp_probs_b.bias", rng.standard_normal(NE).astype(np.float32))
    for l in (0, 1):
        w2.add_tensor("blk.%d.attn_norm.weight" % l, rng.standard_normal(H).astype(np.float32))
        w2.add_tensor("blk.%d.ffn_norm.weight" % l, rng.standard_normal(H).astype(np.float32))
    q8 = quants.quantize(rng.standard_normal(H * 512).astype(np.float32), Q.Q8_0)
    w2.add_tensor("blk.1.attn_kv.weight", q8.reshape(512, H * 34 // 32),
                  raw_shape=[512, H * 34 // 32], raw_dtype=Q.Q8_0)
    w2.add_tensor("output_norm.weight", rng.standard_normal(H).astype(np.float32))
    w2.write_header_to_file()
    w2.write_kv_data_to_file()
    w2.write_tensors_to_file()
    w2.close()
    return s1, s2


class Dsv4PackTests(unittest.TestCase):
    def setUp(self):
        self._tables = (iq_pack.FORM, iq_pack.NOT_IN_PACK, iq_pack.NATIVE_PLE_KEY)

    def tearDown(self):
        iq_pack.FORM, iq_pack.NOT_IN_PACK, iq_pack.NATIVE_PLE_KEY = self._tables

    def _pack(self, root, s1):
        out = root / "pack"
        argv = [sys.argv[0], "--gguf", str(s1), "--out", str(out), "--experts-bin"]
        with contextlib.redirect_stdout(io.StringIO()) as log, patch.object(sys, "argv", argv):
            rc = iq_pack.main()
        self.assertEqual(rc, 0, log.getvalue())
        return out, log.getvalue()

    def test_deepseek4_pack(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            s1, s2 = _write_fixture(root)
            out, log = self._pack(root, s1)

            # native_experts.txt: the completion marker, one line per layer, IQ2_XXS gate/up + IQ3_XXS down,
            # every role in shard 2 (named in the trailing column because --gguf pointed at shard 1)
            lines = [l for l in (out / "native_experts.txt").read_text().splitlines() if not l.startswith("#")]
            self.assertEqual(len(lines), 2)
            for l, line in enumerate(lines):
                f = line.split()
                self.assertEqual([int(f[0]), int(f[1]), int(f[2])], [l, 16, 18])   # IQ2_XXS=16, IQ3_XXS=18
                self.assertEqual(int(f[4]), BLOB)
                self.assertEqual(f[8], s2.name)
            self.assertIn("model.json", log)
            self.assertIn("not exported for deepseek4 yet", log)

            # the arena is the source bytes, relaid as [gate | up | down] per expert
            arena = (out / "experts.bin").read_bytes()
            self.assertEqual(len(arena), 2 * NE * BLOB)
            g2 = iq_pack.G.GGUFFile(s2)
            mm = np.memmap(s2, dtype=np.uint8, mode="r")
            for l in range(2):
                roles = [mm[g2.data_start + t.offset: g2.data_start + t.offset + t.expected_bytes()]
                         for t in g2.tensors if t.name.startswith("blk.%d.ffn_" % l) and t.name.endswith("_exps.weight")]
                for e in range(NE):
                    want = b"".join(bytes(r[e * len(r) // NE: (e + 1) * len(r) // NE]) for r in roles)
                    self.assertEqual(arena[(l * NE + e) * BLOB: (l * NE + e + 1) * BLOB], want,
                                     "layer %d expert %d" % (l, e))

            # index.txt: floats as stored, quantized projections and the I32 hash table served natively
            _, rows = iq_pack.read_index(out / "index.txt")
            dense = (out / "dense.bin").read_bytes()
            for name in ("blk.0.attn_norm.weight", "blk.1.attn_norm.weight", "blk.1.exp_probs_b.bias",
                         "output_norm.weight"):
                self.assertEqual(rows[name][2], "2", name)
            self.assertEqual(rows["blk.1.ffn_gate_inp.weight"][2], "4")             # BF16 router, as stored
            for name in ("token_embd.weight", "output.weight", "blk.1.attn_kv.weight",
                         "blk.0.ffn_gate_tid2eid.weight"):
                self.assertEqual(rows[name][2], "0", name)                          # native row
                self.assertEqual(rows[name][9], "8", name)
            r = rows["blk.0.attn_norm.weight"]
            off, size = int(r[3]), int(r[4])
            t = next(t for t in g2.tensors if t.name == "blk.0.attn_norm.weight")
            self.assertEqual(dense[off:off + size],
                             mm[g2.data_start + t.offset: g2.data_start + t.offset + t.expected_bytes()].tobytes())

            # model.json: the geometry P2 reads, without the heavy tokenizer arrays
            man = json.loads((out / "model.json").read_text())
            self.assertEqual(man["architecture"], "deepseek4")
            self.assertEqual(man["metadata"]["deepseek4.block_count"], 2)
            self.assertEqual(man["metadata"]["deepseek4.attention.compress_ratios"], [0, 4])
            self.assertNotIn("tokenizer.ggml.tokens", man["metadata"])

    def test_hash_layer_without_router_still_packs_expert_count_from_tensors(self):
        # the fixture's layer 0 has no ffn_gate_inp; the count must come from the expert tensors
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            s1, s2 = _write_fixture(root)
            model = iq_pack.Model(s1)
            got = iq_pack.expert_layout(model, s1)
            self.assertFalse(isinstance(got, str))
            self.assertEqual(got[2], NE)


if __name__ == "__main__":
    unittest.main()
