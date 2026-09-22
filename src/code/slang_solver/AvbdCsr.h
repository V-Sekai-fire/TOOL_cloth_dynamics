// Vertex -> incident-constraint CSR, shared by the AVBD backends.
//
// This is the C++ side of `Cloth.Avbd.AdjacencyKwise.CsrAdj.build` and
// `Cloth.Avbd.AdjacencySpring`. The gather kernels index their
// per-constraint inputs as [K*c + role], so `role` has to carry the
// vertex's position within its constraint's incidence list or every
// gather reads the wrong slot.
//
// Lifted out of AvbdSolverVk.cpp so test_avbd_csr.cpp can pin it
// against the Lean fixtures rather than trusting a reimplementation:
// the Lean side proves its own build with `native_decide`, so matching
// its output on the same fixtures is the strongest check available
// without porting the proof.
//
//   offsets[v .. v+1)  slice into cIdx / role for vertex v
//   cIdx[k]            constraint id of the k-th incidence at vertex v
//   role[k]            position of v within that constraint (0 .. K-1)

#ifndef CLOTH_AVBD_CSR_H
#define CLOTH_AVBD_CSR_H

#include <cstdint>
#include <vector>

namespace cloth {

// Build the CSR for `nC` constraints of arity `k`, whose vertex indices
// are `idx[k*c + r]`.
//
// `vcap` sizes the offsets array as `vcap + 1` rather than `nV + 1`.
// The Vulkan backend allocates every buffer to a padded capacity so a
// ragged final workgroup lands on a dummy slot instead of running off
// the end (no AVBD kernel bounds-checks its thread id), and those
// padded vertices must gather an empty range. Pass `vcap == nV` for the
// exact shape the Lean spec describes.
inline void buildCsr(uint32_t nV, uint32_t vcap, const uint32_t *idx, uint32_t nC,
		uint32_t k, std::vector<uint32_t> &offsets, std::vector<uint32_t> &cIdx,
		std::vector<uint32_t> &role) {
	std::vector<uint32_t> counts(size_t(nV) + 1, 0u);
	for (uint32_t c = 0; c < nC; ++c) {
		for (uint32_t r = 0; r < k; ++r) counts[idx[k * c + r]]++;
	}

	offsets.assign(size_t(vcap) + 1, 0u);
	for (uint32_t v = 0; v < nV; ++v) offsets[v + 1] = offsets[v] + counts[v];
	const uint32_t total = offsets[nV];
	// The dummy vertex and every padded slot gather nothing.
	for (uint32_t v = nV; v <= vcap; ++v) offsets[v] = total;

	cIdx.assign(total, 0u);
	role.assign(total, 0u);
	std::vector<uint32_t> cursor(size_t(nV) + 1, 0u);
	for (uint32_t c = 0; c < nC; ++c) {
		for (uint32_t r = 0; r < k; ++r) {
			const uint32_t v = idx[k * c + r];
			const uint32_t p = offsets[v] + cursor[v]++;
			cIdx[p] = c;
			role[p] = r;
		}
	}
}

}  // namespace cloth

#endif  // CLOTH_AVBD_CSR_H
