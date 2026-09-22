// Multi-color conformance test for the Vulkan AVBD backend.
//
// test_avbd_solver.cpp covers the single-color (block Jacobi) path: it
// never calls buildColoring(), so numColors == 1 and vertPerm is the
// identity. That leaves the two things most likely to break in a
// Metal -> Vulkan port completely untested:
//
//   * ragged tails. vkCmdDispatch takes group counts, so a color whose
//     size is not a multiple of 64 runs extra lanes, and no AVBD kernel
//     bounds-checks tid. The backend pads each color's vertPerm range
//     with a sentinel vertex; if that padding is wrong, overrun lanes
//     write into the NEXT color's vertices.
//
//   * barriers. The Metal backend relies on MTLComputeCommandEncoder's
//     implicit serial ordering and contains no explicit barrier. Vulkan
//     guarantees nothing, so the backend emits one after every dispatch.
//     A missing barrier corrupts results nondeterministically rather
//     than failing outright.
//
// The trick that makes this checkable: replicate the canonical
// 4-vertex fixture M times over disjoint vertex ranges. The replicas
// share no constraint, so nothing the solver does to one may affect
// another -- whatever answer replica 0 produces, every other replica
// must produce bit-for-bit.
//
// Note what can NOT be asserted here. Coloring gives the solve real
// Gauss-Seidel semantics: the four vertices of a replica form a
// 4-clique (the bending stencil), so they land in four different
// colors and v1 sees v0 already updated. That is a different fixed
// point from the single-color Jacobi sweep, so the closed-form values
// in test_avbd_solver.cpp apply only to the uncolored path. This test
// therefore checks the closed form with coloring off, and
// replica-invariance plus determinism with it on.
//
// M = 37 (148 vertices) is deliberately not a multiple of 64, so the
// final workgroup of every dispatch is ragged in both runs.
//
// Usage: test_avbd_vk_colored <spirv-dir>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "AvbdSolver.h"

namespace {

constexpr uint32_t kReplicas = 37;
constexpr uint32_t kVertsPer = 4;
constexpr float kTol = 1e-5f;

// Closed-form result for one replica, from test_avbd_solver.cpp.
const float kExpected[12] = {
	7.0f / 8.0f, 15.0f / 7.0f, 12.0f / 7.0f,  // v0
	12.0f / 5.0f, 1.0f, 0.0f,                 // v1
	0.0f, 25.0f / 8.0f, 5.0f,                 // v2
	3.0f, 8.0f / 3.0f, 0.0f,                  // v3
};

struct Fixture {
	std::vector<float> positions, predicted, mass;
	std::vector<uint32_t> springP1, springP2;
	std::vector<float> springRest, springK;
	std::vector<uint32_t> attachVert;
	std::vector<float> attachFixed, attachK;
	std::vector<uint32_t> triIdx;
	std::vector<float> triInvUV, triK;
	std::vector<uint32_t> bendIdx;
	std::vector<float> bendWeight, bendNTarget, bendK;
};

Fixture build(uint32_t replicas) {
	Fixture f;
	const float pos[12] = {0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0};
	const float pred[12] = {1, 2, 3, 3, 0, 0, 0, 4, 0, 3, 4, 0};
	const float m[4] = {2, 1, 1, 1};

	for (uint32_t r = 0; r < replicas; ++r) {
		const uint32_t base = r * kVertsPer;
		for (int i = 0; i < 12; ++i) {
			f.positions.push_back(pos[i]);
			f.predicted.push_back(pred[i]);
		}
		for (int i = 0; i < 4; ++i) f.mass.push_back(m[i]);

		f.springP1.push_back(base + 0);
		f.springP2.push_back(base + 1);
		f.springRest.push_back(2.0f);
		f.springK.push_back(1.0f);

		f.attachVert.push_back(base + 2);
		f.attachFixed.push_back(0.0f);
		f.attachFixed.push_back(4.0f);
		f.attachFixed.push_back(10.0f);
		f.attachK.push_back(4.0f);

		f.triIdx.push_back(base + 0);
		f.triIdx.push_back(base + 1);
		f.triIdx.push_back(base + 2);
		const float uv[4] = {1, 0, 0, 1};
		for (int i = 0; i < 4; ++i) f.triInvUV.push_back(uv[i]);
		f.triK.push_back(1.0f);

		for (uint32_t i = 0; i < 4; ++i) f.bendIdx.push_back(base + i);
		const float w[4] = {1, 1, -1, -1};
		for (int i = 0; i < 4; ++i) f.bendWeight.push_back(w[i]);
		f.bendNTarget.push_back(4.0f);
		f.bendK.push_back(1.0f);
	}
	return f;
}

void upload(cloth::AvbdSolver &s, const Fixture &f, uint32_t nVerts) {
	s.setupMesh(nVerts, f.positions.data(), f.predicted.data(), f.mass.data(), 2.0f);
	s.uploadSprings(uint32_t(f.springK.size()), f.springP1.data(), f.springP2.data(),
			f.springRest.data(), f.springK.data());
	s.uploadAttachments(uint32_t(f.attachK.size()), f.attachVert.data(),
			f.attachFixed.data(), f.attachK.data());
	s.uploadTriangles(uint32_t(f.triK.size()), f.triIdx.data(), f.triInvUV.data(),
			f.triK.data());
	s.uploadBendings(uint32_t(f.bendK.size()), f.bendIdx.data(), f.bendWeight.data(),
			f.bendNTarget.data(), f.bendK.data());
}

}  // namespace

int main(int argc, char **argv) {
	const char *dir = (argc > 1) ? argv[1] : ".";
	const uint32_t nVerts = kReplicas * kVertsPer;

	const Fixture f = build(kReplicas);
	std::vector<float> out;

	// ---- Pass 1: coloring OFF. Jacobi semantics, so the closed-form
	// values apply to every replica. 148 verts is already ragged, so
	// this alone exercises the sentinel padding on a single color.
	{
		cloth::AvbdSolver solver(dir);
		if (!solver.ok()) {
			std::printf("test_avbd_vk_colored: construction failed\n");
			return 1;
		}
		upload(solver, f, nVerts);
		if (solver.step() != 0) {
			std::printf("test_avbd_vk_colored: uncolored step() failed\n");
			return 1;
		}
		solver.readPositions(out);
		if (out.size() != size_t(3) * nVerts) {
			std::printf("test_avbd_vk_colored: readPositions returned %zu, expected %u\n",
					out.size(), 3 * nVerts);
			return 1;
		}

		int failures = 0;
		float maxDiff = 0.0f;
		for (uint32_t r = 0; r < kReplicas; ++r) {
			for (int i = 0; i < 12; ++i) {
				const float got = out[size_t(r) * 12 + i];
				const float diff = std::fabs(got - kExpected[i]);
				if (diff > maxDiff) maxDiff = diff;
				if (!(diff <= kTol)) {
					if (failures < 5) {
						std::printf(
								"  MISMATCH replica %u comp %d: got %.9g expected %.9g\n",
								r, i, got, kExpected[i]);
					}
					++failures;
				}
			}
		}
		if (failures) {
			std::printf(
					"test_avbd_vk_colored: FAILED (uncolored) -- %d/%u components off, "
					"max_abs_diff=%g\n",
					failures, 12 * kReplicas, maxDiff);
			return 1;
		}
		std::printf(
				"test_avbd_vk_colored: uncolored OK -- %u replicas / %u verts "
				"(ragged tail), max_abs_diff=%g\n",
				kReplicas, nVerts, maxDiff);
	}

	// ---- Pass 2: coloring ON. Gauss-Seidel, so no closed form -- but
	// the replicas are mutually independent, so they must agree with
	// each other exactly, and the whole solve must be deterministic.
	{
		cloth::AvbdSolver solver(dir);
		if (!solver.ok()) {
			std::printf("test_avbd_vk_colored: construction failed (colored)\n");
			return 1;
		}
		upload(solver, f, nVerts);
		solver.buildColoring();

		if (solver.step() != 0) {
			std::printf("test_avbd_vk_colored: colored step() failed\n");
			return 1;
		}
		solver.readPositions(out);

		// Replica invariance: a padding overrun or a cross-color write
		// perturbs some replicas and not others, so this catches it.
		int failures = 0;
		float maxDiff = 0.0f;
		for (uint32_t r = 1; r < kReplicas; ++r) {
			for (int i = 0; i < 12; ++i) {
				const float got = out[size_t(r) * 12 + i];
				const float ref = out[i];
				const float diff = std::fabs(got - ref);
				if (diff > maxDiff) maxDiff = diff;
				if (!(diff <= kTol)) {
					if (failures < 5) {
						std::printf(
								"  REPLICA DRIFT r=%u comp %d: got %.9g, replica0 %.9g "
								"-- independent replicas must agree\n",
								r, i, got, ref);
					}
					++failures;
				}
			}
		}
		if (failures) {
			std::printf(
					"test_avbd_vk_colored: FAILED (colored) -- %d components drifted, "
					"max_abs_diff=%g\n",
					failures, maxDiff);
			return 1;
		}

		// Determinism: a missing barrier is a race, so it shows up as
		// run-to-run drift rather than a stable wrong answer.
		const std::vector<float> first = out;
		for (int trial = 0; trial < 8; ++trial) {
			solver.updateState(f.positions.data(), f.predicted.data());
			if (solver.step() != 0) {
				std::printf("test_avbd_vk_colored: rerun %d failed\n", trial);
				return 1;
			}
			solver.readPositions(out);
			if (std::memcmp(first.data(), out.data(), first.size() * sizeof(float)) != 0) {
				size_t where = 0;
				while (where < out.size() && out[where] == first[where]) ++where;
				std::printf(
						"test_avbd_vk_colored: NON-DETERMINISTIC on rerun %d "
						"(first divergence at component %zu: %.9g vs %.9g) -- "
						"this is what a missing pipeline barrier looks like\n",
						trial, where, first[where], out[where]);
				return 1;
			}
		}
		std::printf(
				"test_avbd_vk_colored: colored OK -- replicas agree to %g, "
				"bit-identical across 8 reruns\n",
				maxDiff);
	}

	std::printf("test_avbd_vk_colored: OK\n");
	return 0;
}
