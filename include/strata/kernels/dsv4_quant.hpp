// include/strata/kernels/dsv4_quant.hpp - the DSv4 pool round-trips and the indexer rotation,
// docs/DSV4.md P2.
//
// DSv4 stores pool rows at reduced precision WITHOUT storing a packed format: `act_quant_fp8_inplace`
// and `fp4_act_quant_inplace` are QUANT+DEQUANT round-trips - the value that goes into the pool is a
// bf16 that happens to sit on the e4m3 / e2m1 grid at a power-of-two scale.  The pool stays a plain
// bf16 array (the gather and indexer kernels read it as such); the precision loss is baked into the
// values.  Packing them is a later memory win, not a correctness need.
//
// The scale convention is the reference's (`fp8_linear.py`): `s = 2 ** ceil(log2(amax / max_value))`
// with `max_value` 448 (e4m3) or 6 (e2m1), an `amax` floor of `1e-4` / `6 * 2**-126`, and the
// quantized value clamped to `+-max_value` before the grid rounding.  The e2m1 grid is
// {0, .5, 1, 1.5, 2, 3, 4, 6} with the hardware's ties-to-even midpoints (0.25->0, 0.75->1,
// 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5.0->4) - the odd-magnitude midpoints round UP to the even grid
// point, the even-magnitude ones round DOWN to it.
//
// `hadamard` is the indexer's `rotate_activation`: the normalized Sylvester Walsh-Hadamard transform
// on the last (power-of-two) dim, `WHT(x) * d**-0.5` computed in fp32 - applied to the index query
// and the rotating compressor keys BEFORE the fp4 round-trip, which is what makes the quantization
// spread the energy instead of concentrating it in a few coordinates.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// In-place-in-value: `y[i] = dequant(quant_e4m3(x[i] / s) * s)` with `s` the power-of-two scale of
/// the `block`-wide group containing `i`.  `n % block == 0`; groups are contiguous.  Output is bf16
/// bits holding e4m3-grid values.
void roundtrip_fp8_e4m3(const float* x, uint16_t* y, int64_t n, int64_t block, void* stream);

/// The same for the e2m1 grid (block 32 in practice).
void roundtrip_fp4_e2m1(const float* x, uint16_t* y, int64_t n, int64_t block, void* stream);

/// `y[row] = WHT(x[row]) * d ** -0.5`, Sylvester order, fp32 in and out, `d` a power of two <= 1024.
void hadamard(const float* x, float* y, int64_t rows, int64_t d, void* stream);

}  // namespace strata::kernels
