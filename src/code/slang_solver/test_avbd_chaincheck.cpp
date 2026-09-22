// How far does AVBD's gradient survive back-propagation through time?
//
// test_avbd_gradcheck.cpp established that the SINGLE-step adjoint is
// correct for the parameters the dress demo uses (k_tri, k_bend,
// k_attach, restLen all match finite differences to ~1e-4). Yet the
// dress stalls: loss improves 0.002% and then eleven evaluations
// return the identical value. So the suspect is the chain, not the
// derivative.
//
// Simulation.cpp's own note says the per-step Jacobian has spectral
// radius ~sqrt(2), so 125 steps amplify ~2^200x; AVBD_BWD_TRUNCATE_K=20
// clips the chain to bound that, which makes the result a degree-20
// Neumann approximation rather than the gradient. Nobody has measured
// what that costs.
//
// This measures it on 4 vertices instead of 3634, by sweeping the
// number of chained steps N and comparing the accumulated analytic
// gradient against a finite difference of the N-step loss. Seconds per
// sweep. The step count at which the relative error crosses out of
// usefulness is the answer to "can PD's adjoint be retired".
//
// The predictor here is deliberately x_pred = x_prev (quasi-static, no
// velocity term), so the chain is exactly N applications of the
// per-step Jacobian and dL/dx_{k-1} = positionsGrad + predictedGrad.
// That isolates the amplification question from the integrator.
//
// Build:
//   clang++ -std=c++17 -I<vulkan>/Include -I. \
//     test_avbd_chaincheck.cpp AvbdSolverVk.cpp -lvulkan-1

#include <cmath>
#include <cstdio>
#include <vector>

#include "AvbdSolver.h"

namespace {

constexpr uint32_t N = 4;
const float kPos0[12] = {0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0};
const float kMass[4] = {2, 1, 1, 1};

void uploadAll(cloth::AvbdSolver &s, const std::vector<float> &pos,
		const std::vector<float> &pred, float triK) {
	s.setupMesh(N, pos.data(), pred.data(), kMass, 2.0f);
	const uint32_t p1[1] = {0}, p2[1] = {1};
	const float rl[1] = {2.0f}, sk[1] = {1.0f};
	s.uploadSprings(1, p1, p2, rl, sk);
	const uint32_t av[1] = {2};
	const float af[3] = {0, 4, 10}, ak[1] = {4.0f};
	s.uploadAttachments(1, av, af, ak);
	const uint32_t ti[3] = {0, 1, 2};
	const float uv[4] = {1, 0, 0, 1}, tk[1] = {triK};
	s.uploadTriangles(1, ti, uv, tk);
	const uint32_t bi[4] = {0, 1, 2, 3};
	const float bw[4] = {1, 1, -1, -1}, bn[1] = {4}, bk[1] = {1.0f};
	s.uploadBendings(1, bi, bw, bn, bk);
}

// Run `steps` chained steps; return 0.5*|x_final|^2. Optionally record
// the per-step input states the backward pass needs to replay.
double forwardChain(cloth::AvbdSolver &s, float triK, int steps,
		std::vector<std::vector<float>> *states = nullptr) {
	std::vector<float> pos(kPos0, kPos0 + 12);
	if (states) states->clear();
	for (int k = 0; k < steps; ++k) {
		if (states) states->push_back(pos);
		// Quasi-static predictor: the predictor IS the previous state.
		uploadAll(s, pos, pos, triK);
		if (s.step() != 0) return NAN;
		std::vector<float> nxt;
		s.readPositions(nxt);
		if (nxt.size() != size_t(N) * 3) return NAN;
		pos = nxt;
	}
	double L = 0.0;
	for (float v : pos) L += 0.5 * double(v) * double(v);
	return L;
}

// Accumulated dL/d triK over the chain, by replaying each step and
// propagating the position cotangent backwards.
double analyticChain(cloth::AvbdSolver &s, float triK, int steps) {
	std::vector<std::vector<float>> states;
	const double L = forwardChain(s, triK, steps, &states);
	if (!(L == L)) return NAN;

	// Seed: L = 0.5|x_N|^2 so dL/dx_N = x_N. Recompute the final state.
	std::vector<float> pos = states.back();
	uploadAll(s, pos, pos, triK);
	if (s.step() != 0) return NAN;
	std::vector<float> adj;
	s.readPositions(adj);

	double dL_dk = 0.0;
	for (int k = steps - 1; k >= 0; --k) {
		// Replay step k so the solver holds that step's forward state.
		uploadAll(s, states[k], states[k], triK);
		if (s.step() != 0) return NAN;
		if (s.stepBackward(adj.data()) != 0) return NAN;

		std::vector<float> dTriK, dL0, dL1;
		s.readTriGrad(dTriK, dL0, dL1);
		if (!dTriK.empty()) dL_dk += double(dTriK[0]);

		// dL/dx_{k-1} = positionsGrad + predictedGrad, since the
		// predictor is the previous state itself.
		std::vector<float> gPos, gPred;
		s.readPositionsGrad(gPos);
		s.readPredictedGrad(gPred);
		adj.assign(size_t(N) * 3, 0.0f);
		for (size_t i = 0; i < adj.size(); ++i) {
			const double a = (i < gPos.size()) ? double(gPos[i]) : 0.0;
			const double b = (i < gPred.size()) ? double(gPred[i]) : 0.0;
			adj[i] = float(a + b);
		}
	}
	return dL_dk;
}

}  // namespace

int main(int argc, char **argv) {
	const char *dir = (argc > 1) ? argv[1] : ".";
	cloth::AvbdSolver solver(dir);
	if (!solver.ok()) {
		std::printf("test_avbd_chaincheck: construction failed\n");
		return 1;
	}

	const float triK = 1.0f;
	const double h = 1e-3;

	std::printf("test_avbd_chaincheck: dL/d k_tri accumulated over N chained steps\n");
	std::printf("  4 vertices, quasi-static predictor, central differences\n\n");
	std::printf("  %4s %14s %14s %11s  %s\n", "N", "analytic", "fd", "rel", "verdict");
	std::printf("  ---------------------------------------------------------------\n");

	int firstBad = 0;
	const int steps[] = {1, 2, 4, 8, 16, 32};
	for (int n : steps) {
		const double a = analyticChain(solver, triK, n);
		const double fd =
				(forwardChain(solver, float(triK + h), n) -
						forwardChain(solver, float(triK - h), n)) /
				(2.0 * h);
		const double denom = std::max(1.0, std::max(std::fabs(a), std::fabs(fd)));
		const double rel = std::fabs(a - fd) / denom;
		const char *verdict = (rel < 0.05) ? "ok" : (rel < 0.5) ? "degraded" : "USELESS";
		if (rel >= 0.05 && firstBad == 0) firstBad = n;
		std::printf("  %4d %14.6g %14.6g %11.3g  %s\n", n, a, fd, rel, verdict);
	}

	std::printf("\n");
	if (firstBad) {
		std::printf("test_avbd_chaincheck: the chained gradient departs from finite\n"
					"differences by N = %d steps. The per-step adjoint is correct\n"
					"(test_avbd_gradcheck), so this is the accumulation itself --\n"
					"which is what AVBD_BWD_TRUNCATE_K is papering over, and what\n"
					"has to be fixed before PD's adjoint can be retired.\n",
				firstBad);
		return 0;  // a measurement, not a pass/fail gate
	}
	std::printf("test_avbd_chaincheck: the chain holds to 32 steps. BPTT\n"
				"amplification is NOT the dress stall's cause; look elsewhere.\n");
	return 0;
}
