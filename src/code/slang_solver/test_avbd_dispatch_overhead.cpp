// How much of AVBD's per-step cost is GPU round-trip overhead?
//
// The sphere demo measures 31.45 ms/step under AVBD against 8.5 ms/step
// for the old CPU Projective-Dynamics solver -- AVBD is ~4x SLOWER on a
// ~600-vertex mesh. The suspicion is that this is not arithmetic at all:
// AvbdSolver::step() does one vkQueueSubmit followed by a fence wait,
// and Simulation calls step() once per outer iteration, 16 times per
// timestep. That is 16 submit+wait round-trips per simulated step.
//
// If that is right, per-step cost should be nearly INDEPENDENT of mesh
// size over the range where the GPU is not saturated: a 4-vertex mesh
// and a 5000-vertex mesh would cost about the same, because both are
// paying for the round-trip rather than the work. If instead cost scales
// with vertex count, the time is real compute and a CPU port is the
// interesting lead rather than batching.
//
// The test sweeps mesh size and reports microseconds per step() call.
// A flat column is the diagnosis; a rising one refutes it.
//
// Build:
//   clang++ -std=c++17 -O2 -I<vulkan>/Include -I. \
//     test_avbd_dispatch_overhead.cpp AvbdSolverVk.cpp -lvulkan-1

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "AvbdSolver.h"

namespace {

// A w x h grid of vertices, triangulated, with every triangle a membrane
// constraint -- the same constraint mix the cloth demos use.
struct Grid {
	uint32_t nVerts = 0;
	std::vector<float> pos, mass;
	std::vector<uint32_t> triIdx;
	std::vector<float> triUV, triK;

	Grid(int w, int h) {
		nVerts = uint32_t(w * h);
		pos.resize(3 * size_t(nVerts));
		mass.assign(nVerts, 1.0f);
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				const size_t v = size_t(y) * w + x;
				pos[3 * v + 0] = float(x) * 0.1f;
				pos[3 * v + 1] = float(y) * 0.1f;
				pos[3 * v + 2] = 0.0f;
			}
		}
		for (int y = 0; y + 1 < h; ++y) {
			for (int x = 0; x + 1 < w; ++x) {
				const uint32_t a = uint32_t(y * w + x), b = a + 1;
				const uint32_t c = uint32_t((y + 1) * w + x), d = c + 1;
				addTri(a, b, c);
				addTri(b, d, c);
			}
		}
		triK.assign(triIdx.size() / 3, 10.0f);
	}
	void addTri(uint32_t a, uint32_t b, uint32_t c) {
		triIdx.push_back(a);
		triIdx.push_back(b);
		triIdx.push_back(c);
		// Identity rest material: enough to exercise the kernel.
		triUV.push_back(1.0f);
		triUV.push_back(0.0f);
		triUV.push_back(0.0f);
		triUV.push_back(1.0f);
	}
	uint32_t nTri() const { return uint32_t(triK.size()); }
};

double timeSteps(cloth::AvbdSolver &s, const Grid &g, int iters, int reps,
		bool coloring) {
	std::vector<float> pred = g.pos;
	for (uint32_t v = 0; v < g.nVerts; ++v) pred[3 * v + 2] -= 0.001f;

	s.setupMesh(g.nVerts, g.pos.data(), pred.data(), g.mass.data(), 3600.0f);
	s.uploadSprings(0, nullptr, nullptr, nullptr, nullptr);
	s.uploadAttachments(0, nullptr, nullptr, nullptr);
	s.uploadTriangles(g.nTri(), g.triIdx.data(), g.triUV.data(), g.triK.data());
	s.uploadBendings(0, nullptr, nullptr, nullptr, nullptr);
	if (coloring) s.buildColoring();

	// Warm up: first submit pays for pipeline and descriptor setup.
	for (int i = 0; i < 3; ++i) s.step();

	const auto t0 = std::chrono::steady_clock::now();
	for (int r = 0; r < reps; ++r) {
		for (int i = 0; i < iters; ++i) {
			if (s.step() != 0) return -1.0;
		}
	}
	const auto t1 = std::chrono::steady_clock::now();
	const double us =
			double(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
	return us / double(reps);
}

}  // namespace

int main(int argc, char **argv) {
	setvbuf(stdout, nullptr, _IONBF, 0);
	const char *dir = (argc > 1) ? argv[1] : ".";
	cloth::AvbdSolver solver(dir);
	if (!solver.ok()) {
		std::printf("test_avbd_dispatch_overhead: construction failed\n");
		return 1;
	}

	std::printf("test_avbd_dispatch_overhead: us per simulated step\n");
	std::printf("  (16 AVBD outer iterations = 16 submit+fence round-trips)\n\n");
	std::printf("  %7s %8s %10s %14s %14s\n", "verts", "tris", "colors", "us/step",
			"us/iteration");

	const int sizes[] = {2, 4, 8, 16, 24, 32, 48, 64};
	double firstPerIter = 0.0, lastPerIter = 0.0;
	uint32_t firstV = 0, lastV = 0;
	for (int n : sizes) {
		Grid g(n, n);
		const double us = timeSteps(solver, g, 16, 20, true);
		if (us < 0) {
			std::printf("  %7u   step failed\n", g.nVerts);
			return 1;
		}
		std::printf("  %7u %8u %10s %14.1f %14.1f\n", g.nVerts, g.nTri(), "auto", us,
				us / 16.0);
		if (firstPerIter == 0.0) {
			firstPerIter = us / 16.0;
			firstV = g.nVerts;
		}
		lastPerIter = us / 16.0;
		lastV = g.nVerts;
	}

	const double vertRatio = double(lastV) / double(firstV);
	const double costRatio = lastPerIter / (firstPerIter > 0 ? firstPerIter : 1.0);
	std::printf("\n  vertices grew %.0fx; cost per iteration grew %.2fx\n", vertRatio,
			costRatio);

	// If cost is dominated by the round-trip, a 1000x larger mesh costs
	// only marginally more. Call it overhead-bound if cost grew less than
	// a tenth as fast as the mesh.
	if (costRatio < vertRatio * 0.1) {
		std::printf("\n  => There is a large FIXED cost per iteration that does not\n"
					"     depend on mesh size. That is the vkQueueSubmit + fence\n"
					"     round-trip, not arithmetic. Batching the outer iterations\n"
					"     into one command buffer removes all but one of them.\n");
		std::printf("\n  Measured floor is the smallest us/iteration above. At 16\n"
					"  iterations per simulated step that floor is paid 16 times,\n"
					"  so it sets a hard lower bound on step cost no amount of\n"
					"  kernel tuning can go below.\n");
		std::printf("\n  Note what this does NOT say. Removing the round-trip does not\n"
					"  by itself make AVBD win on a small mesh -- on the sphere the\n"
					"  floor is a bit over half the solve time, and the old CPU\n"
					"  solver is ~4x faster there overall. A CPU AVBD backend\n"
					"  remains a legitimate lead for small meshes; this only says\n"
					"  batching is the cheaper thing to try first.\n");
	} else {
		std::printf("\n  => COMPUTE-BOUND. Cost tracks mesh size, so the per-step time\n"
					"     is real work. Batching would not help much; a CPU backend is\n"
					"     the lead worth following for small meshes.\n");
	}
	return 0;
}
