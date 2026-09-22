// Are readPositionsGrad and readPredictedGrad each correct, separately?
//
// test_avbd_chaincheck's state check aliased predicted = x, which is
// why it could not arbitrate. vbd_init_backward emits
//
//     v_x = +w * v_g        v_y = -w * v_g        (w = m * invH^2)
//
// exact negatives, and the Vulkan epilogue folds v_x into
// positionsGrad. So under aliasing, positionsGrad + predictedGrad
// cancels the inertial path entirely, and a disagreement cannot be
// attributed to either accessor -- or to the composition.
//
// This removes the ambiguity by never aliasing. positions and
// predicted start as DIFFERENT vectors, and each is finite-differenced
// on its own with the other held fixed:
//
//     d L / d positions[i]  should equal  readPositionsGrad[i]
//     d L / d predicted[i]  should equal  readPredictedGrad[i]
//
// Two independent verdicts, no composition involved. Whichever
// accessor fails is the one to fix; if both pass, the earlier N=2
// chain break is in how steps are composed, not in the accessors.
//
// Build:
//   clang++ -std=c++17 -I<vulkan>/Include -I. \
//     test_avbd_stategrad.cpp AvbdSolverVk.cpp -lvulkan-1

#include <cmath>
#include <cstdio>
#include <vector>

#include "AvbdSolver.h"

namespace {

constexpr uint32_t NV = 4;
// Deliberately distinct, so no cancellation can hide a fault.
const float kPos0[12] = {0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0};
const float kPred0[12] = {1, 2, 3, 3, 0, 0, 0, 4, 0, 3, 4, 0};
const float kMass[4] = {2, 1, 1, 1};

int g_useSpring = 1, g_useAttach = 1, g_useTri = 1, g_useBend = 1;

void uploadAll(cloth::AvbdSolver &s, const float *pos, const float *pred) {
	s.setupMesh(NV, pos, pred, kMass, 2.0f);
	const uint32_t p1[1] = {0}, p2[1] = {1};
	const float rl[1] = {2.0f}, sk[1] = {1.0f};
	s.uploadSprings(g_useSpring ? 1 : 0, p1, p2, rl, sk);
	const uint32_t av[1] = {2};
	const float af[3] = {0, 4, 10}, ak[1] = {4.0f};
	s.uploadAttachments(g_useAttach ? 1 : 0, av, af, ak);
	const uint32_t ti[3] = {0, 1, 2};
	const float uv[4] = {1, 0, 0, 1}, tk[1] = {1.0f};
	s.uploadTriangles(g_useTri ? 1 : 0, ti, uv, tk);
	const uint32_t bi[4] = {0, 1, 2, 3};
	const float bw[4] = {1, 1, -1, -1}, bn[1] = {4}, bk[1] = {1.0f};
	s.uploadBendings(g_useBend ? 1 : 0, bi, bw, bn, bk);
}

// One step from (pos, pred); returns 0.5 * |x_out|^2.
double loss(cloth::AvbdSolver &s, const float *pos, const float *pred) {
	uploadAll(s, pos, pred);
	if (s.step() != 0) return NAN;
	std::vector<float> x;
	s.readPositions(x);
	if (x.size() != size_t(NV) * 3) return NAN;
	double L = 0.0;
	for (float v : x) L += 0.5 * double(v) * double(v);
	return L;
}

// Finite difference of the loss w.r.t. one component of one input,
// with the other input held fixed.
double fd(cloth::AvbdSolver &s, bool wrtPositions, int i, double h) {
	float pHi[12], pLo[12], qHi[12], qLo[12];
	for (int k = 0; k < 12; ++k) {
		pHi[k] = pLo[k] = kPos0[k];
		qHi[k] = qLo[k] = kPred0[k];
	}
	if (wrtPositions) {
		pHi[i] = float(double(kPos0[i]) + h);
		pLo[i] = float(double(kPos0[i]) - h);
	} else {
		qHi[i] = float(double(kPred0[i]) + h);
		qLo[i] = float(double(kPred0[i]) - h);
	}
	return (loss(s, pHi, qHi) - loss(s, pLo, qLo)) / (2.0 * h);
}

int sweep(cloth::AvbdSolver &s, const char *label, bool wrtPositions,
		const std::vector<float> &analytic) {
	std::printf("  %s\n", label);
	std::printf("    %-3s %14s %14s %11s  %s\n", "i", "analytic", "fd", "rel", "");
	int bad = 0;
	const double h = 1e-3;
	for (int i = 0; i < 12; ++i) {
		const double a = (size_t(i) < analytic.size()) ? double(analytic[i]) : 0.0;
		const double n = fd(s, wrtPositions, i, h);
		const double rel = std::fabs(a - n) / std::max(1.0, std::fabs(n));
		if (rel >= 0.05) ++bad;
		std::printf("    %-3d %14.6g %14.6g %11.3g  %s\n", i, a, n, rel,
				rel < 0.05 ? "ok" : "MISMATCH");
	}
	std::printf("    -> %d of 12 disagree\n\n", bad);
	return bad;
}

}  // namespace

int main(int argc, char **argv) {
	const char *dir = (argc > 1) ? argv[1] : ".";
	cloth::AvbdSolver solver(dir);
	if (!solver.ok()) {
		std::printf("test_avbd_stategrad: construction failed\n");
		return 1;
	}

	std::printf("test_avbd_stategrad: positions and predicted checked SEPARATELY\n");
	std::printf("  (never aliased, so neither accessor can hide behind the other)\n\n");

	// --- FALSIFIABILITY -------------------------------------------
	// A sweep that cannot fail proves nothing. Feed it a deliberately
	// corrupted analytic gradient (the true one, doubled) and require
	// it to be rejected. If this ever reports 0 disagreements, the
	// comparison below has lost its teeth and every "ok" is worthless.
	{
		g_useSpring = g_useAttach = g_useTri = g_useBend = 1;
		uploadAll(solver, kPos0, kPred0);
		solver.step();
		std::vector<float> xo;
		solver.readPositions(xo);
		solver.stepBackward(xo.data());
		std::vector<float> gp;
		solver.readPredictedGrad(gp);  // the one known to be correct
		std::vector<float> corrupted = gp;
		for (float &v : corrupted) v *= 2.0f;

		int bad = 0;
		for (int i = 0; i < 12; ++i) {
			const double a = (size_t(i) < corrupted.size()) ? double(corrupted[i]) : 0.0;
			const double n = fd(solver, false, i, 1e-3);
			if (std::fabs(a - n) / std::max(1.0, std::fabs(n)) >= 0.05) ++bad;
		}
		std::printf("  falsifiability: doubled predictedGrad -> %d of 12 rejected %s\n\n",
				bad, bad > 0 ? "(good, the check has teeth)"
							 : "(BAD: the check cannot fail)");
		if (bad == 0) {
			std::printf("test_avbd_stategrad: ABORT -- the comparison cannot detect a\n"
						"doubled gradient, so nothing below can be trusted.\n");
			return 1;
		}
	}

	// --- bisect by constraint family --------------------------------
	{
		const struct { const char *n; int sp, at, tr, be; } cases[] = {
			{"inertia only", 0, 0, 0, 0},
			{"attachment", 0, 1, 0, 0},
			{"spring", 1, 0, 0, 0},
			{"triangle", 0, 0, 1, 0},
			{"bending", 0, 0, 0, 1},
		};
		std::printf("  bisect of d L / d positions by constraint family:\n");
		for (const auto &c : cases) {
			g_useSpring = c.sp; g_useAttach = c.at; g_useTri = c.tr; g_useBend = c.be;
			uploadAll(solver, kPos0, kPred0);
			if (solver.step() != 0) continue;
			std::vector<float> xo;
			solver.readPositions(xo);
			if (solver.stepBackward(xo.data()) != 0) continue;
			std::vector<float> gp;
			solver.readPositionsGrad(gp);
			int bad = 0;
			for (int i = 0; i < 12; ++i) {
				const double a = (size_t(i) < gp.size()) ? double(gp[i]) : 0.0;
				const double n = fd(solver, true, i, 1e-3);
				if (std::fabs(a - n) / std::max(1.0, std::fabs(n)) >= 0.05) ++bad;
			}
			std::printf("    %-14s %d of 12 disagree\n", c.n, bad);
		}
		std::printf("\n");
	}

	// --- full model, both accessors ---------------------------------
	g_useSpring = g_useAttach = g_useTri = g_useBend = 1;
	uploadAll(solver, kPos0, kPred0);
	solver.step();
	std::vector<float> xOut;
	solver.readPositions(xOut);
	solver.stepBackward(xOut.data());
	std::vector<float> gPos, gPred;
	solver.readPositionsGrad(gPos);
	solver.readPredictedGrad(gPred);

	const int badPos = sweep(solver, "d L / d positions  vs readPositionsGrad", true, gPos);
	const int badPred =
			sweep(solver, "d L / d predicted  vs readPredictedGrad", false, gPred);

	if (badPos == 0 && badPred == 0) {
		std::printf("test_avbd_stategrad: BOTH accessors correct; a multi-step break\n"
					"is then in composition, not in the per-step state gradients.\n");
		return 0;
	}
	std::printf("test_avbd_stategrad: positions %d of 12, predicted %d of 12 disagree.\n"
				"See the bisect above for which constraint family is responsible.\n",
			badPos, badPred);
	return 1;
}
