// Does a badly-conditioned membrane freeze the AVBD solve?
//
// On the dress demo the AVBD solve returns |dx| == 0 from step 1 --
// the cloth never moves, every step is byte-identical, and a downstream
// Eigen size assertion fires ~120 steps later. The dress is a terrible
// place to debug that: 3634 verts, 125 steps, minutes per run.
//
// The suspicious difference from a demo that works is triangle
// conditioning. The solver logs its rest-material inverse range:
//
//     sphere (works) : inv_deltaUV max |M| =  5.33
//     dress  (frozen): inv_deltaUV max |M| = 38.58
//
// vbd_solve_apply inverts the accumulated per-vertex 3x3:
//
//     det    = Hxx*a00 + Hxy*a01 + Hxz*a02
//     invDet = 1.0 / det
//     dx     = -invDet * (a00*gx + a01*gy + a02*gz)
//
// The membrane Hessian scales with |inv_deltaUV|^2, so a sliver
// triangle inflates H, det grows as its cube, and invDet collapses.
// In fp32 a large enough |M| drives dx to exactly 0 -- a frozen solve
// that reports rc=0 and nan=0, which is precisely what the dress does.
//
// This sweeps |inv_deltaUV| on the 4-vertex fixture and reports where
// the displacement collapses. Seconds per run instead of minutes.
//
// Build:
//   clang++ -std=c++17 -I<vulkan>/Include -I. \
//     test_avbd_conditioning.cpp AvbdSolverVk.cpp -lvulkan-1

#include <cmath>
#include <cstdio>
#include <vector>

#include "AvbdSolver.h"

namespace {

// The fixture from test_avbd_solver.cpp, with the triangle's rest
// material inverse scaled by `s`. s = 1 is the well-conditioned case.
double stepWithConditioning(cloth::AvbdSolver &solver, float s,
		float invHSq = 2.0f, float kTri = 1.0f, float kBend = 1.0f, float kAttach = 4.0f) {
	const uint32_t N = 4;
	const float positions[12] = {0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0};
	const float predicted[12] = {1, 2, 3, 3, 0, 0, 0, 4, 0, 3, 4, 0};
	const float mass[4] = {2, 1, 1, 1};
	solver.setupMesh(N, positions, predicted, mass, invHSq);

	const uint32_t p1[1] = {0}, p2[1] = {1};
	const float restLen[1] = {2}, springK[1] = {1};
	solver.uploadSprings(1, p1, p2, restLen, springK);

	const uint32_t attachVert[1] = {2};
	const float attachFixed[3] = {0, 4, 10}; const float attachK[1] = {kAttach};
	solver.uploadAttachments(1, attachVert, attachFixed, attachK);

	const uint32_t triIdx[3] = {0, 1, 2};
	// Identity scaled by s: a sliver triangle has large entries here.
	const float invUV[4] = {s, 0, 0, s};
	const float triK[1] = {kTri};
	solver.uploadTriangles(1, triIdx, invUV, triK);

	const uint32_t bendIdx[4] = {0, 1, 2, 3};
	const float bendW[4] = {1, 1, -1, -1};
	const float bendN[1] = {4}; const float bendK[1] = {kBend};
	solver.uploadBendings(1, bendIdx, bendW, bendN, bendK);

	if (solver.step() != 0) return -1.0;

	std::vector<float> out;
	solver.readPositions(out);
	if (out.size() != size_t(N) * 3) return -1.0;

	double dxMax = 0.0;
	for (uint32_t i = 0; i < N * 3; ++i) {
		const double d = std::fabs(double(out[i]) - double(positions[i]));
		if (d > dxMax) dxMax = d;
	}
	return dxMax;
}

}  // namespace


int main(int argc, char **argv) {
	const char *dir = (argc > 1) ? argv[1] : ".";
	cloth::AvbdSolver solver(dir);
	if (!solver.ok()) {
		std::printf("test_avbd_conditioning: construction failed\n");
		return 1;
	}

	struct Case {
		const char *name;
		float m, invHSq, kTri, kBend, kAttach;
	};
	// The fixture baseline, then each dress-like parameter introduced on
	// its own, then all of them together. Whichever line first reads
	// FROZEN names the culprit; if none do, the trigger is not a scalar
	// parameter and the mesh itself has to be the difference.
	const Case cases[] = {
		{"fixture baseline", 1.0f, 2.0f, 1.0f, 1.0f, 4.0f},
		{"dress inv_deltaUV (38.58)", 38.58f, 2.0f, 1.0f, 1.0f, 4.0f},
		{"dress invH^2 (8100, h=1/90)", 1.0f, 8100.0f, 1.0f, 1.0f, 4.0f},
		{"dress membrane k (10000)", 1.0f, 2.0f, 10000.0f, 1.0f, 4.0f},
		{"dress bending k (0)", 1.0f, 2.0f, 1.0f, 0.0f, 4.0f},
		{"membrane k x area (10000*0.0176)", 1.0f, 2.0f, 176.0f, 1.0f, 4.0f},
		{"all dress-like together", 38.58f, 8100.0f, 10000.0f, 0.0f, 4.0f},
	};

	std::printf("  %-34s %13s  %s\n", "case", "|dx|_max", "state");
	std::printf("  --------------------------------------------------------------\n");
	int frozen = 0;
	for (const Case &c : cases) {
		const double dx =
				stepWithConditioning(solver, c.m, c.invHSq, c.kTri, c.kBend, c.kAttach);
		const char *state = (dx < 0.0)    ? "step() FAILED"
							: (dx == 0.0) ? "FROZEN  <-- dx is exactly zero"
							: (dx < 1e-9) ? "collapsing"
										  : "moving";
		if (dx == 0.0) ++frozen;
		std::printf("  %-34s %13.6g  %s\n", c.name, dx, state);
	}

	if (frozen) {
		std::printf("\ntest_avbd_conditioning: %d case(s) froze -- the parameter on\n"
					"that line reproduces the dress symptom on 4 vertices.\n",
				frozen);
	} else {
		std::printf("\ntest_avbd_conditioning: nothing froze. The dress freeze is NOT\n"
					"explained by inv_deltaUV conditioning, timestep, or stiffness in\n"
					"isolation or combination -- so it is a property of that mesh or\n"
					"of the host-side setup, not of these scalars.\n");
	}
	return 0;
}
