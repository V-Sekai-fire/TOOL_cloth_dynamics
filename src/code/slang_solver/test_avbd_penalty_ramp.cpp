// Does AVBD's Eq. 16 penalty ramp actually help?
//
// The augmented-Lagrangian dual update was here already: each outer
// iteration does lambda += gamma * C. What was missing is the other half
// of AVBD, the adaptive penalty itself --
//
//     gamma <- min(gamma + beta * |C|, penaltyMax)             (Eq. 16)
//
// -- which is what lets a constraint stiffen until it is actually
// satisfied instead of sitting at whatever fixed gamma it was handed.
// The review of the authors' reference (savant117/avbd-demo3d,
// Joint::updateDual) found we implemented the lambda ascent and not the
// ramp; this is the test for the half that was added.
//
// The scene is the smallest thing whose convergence depends on it: one
// vertex pinned by a SOFT attachment, pulled by gravity, nothing else.
// A pure penalty of stiffness k settles at a violation of m*g/k and
// stays there. Dual ascent alone drives the violation to zero but at a
// rate fixed by gamma. The ramp raises gamma as the violation persists,
// so the claim under test is specifically about RATE.
//
// Both arms run in one process, differing only in AVBD_BETA, so nothing
// but the ramp separates them:
//
//   beta = 0      the exact previous behaviour, fixed gamma
//   beta = 1e4    upstream's betaLin default
//
// If the two curves agree, the ramp does nothing and should not have
// been added. That is the falsifiable claim.
//
// Build:
//   clang++ -std=c++17 -I<vulkan>/Include -I. \
//     test_avbd_penalty_ramp.cpp AvbdSolverVk.cpp -lvulkan-1

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "AvbdSolver.h"

namespace {

constexpr uint32_t NV = 4;
constexpr float kDt = 1.0f / 60.0f;
constexpr float kGravity = -10.0f;
// Deliberately soft: with k this low a pure penalty sags visibly, so
// there is room for the dual machinery to show what it does.
constexpr float kAttachK = 50.0f;

const float kPos0[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
const float kMass[4] = {1, 1, 1, 1};

void upload(cloth::AvbdSolver &s, const std::vector<float> &pos,
		const std::vector<float> &pred) {
	s.setupMesh(NV, pos.data(), pred.data(), kMass, 1.0f / (kDt * kDt));
	const uint32_t av[1] = {0};
	const float af[3] = {0, 0, 0}, ak[1] = {kAttachK};
	s.uploadAttachments(1, av, af, ak);
	s.uploadSprings(0, nullptr, nullptr, nullptr, nullptr);
	s.uploadTriangles(0, nullptr, nullptr, nullptr);
	s.uploadBendings(0, nullptr, nullptr, nullptr, nullptr);
}

// Run `iters` outer iterations of (primal step, dual update) on one
// step's problem, returning |x_0 - fixedPos| -- the violation of the
// attachment the ramp is supposed to close.
bool run(cloth::AvbdSolver &s, float beta, int iters, double *outViol) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%g", double(beta));
#ifdef _WIN32
	_putenv_s("AVBD_BETA", buf);
#else
	setenv("AVBD_BETA", buf, 1);
#endif

	std::vector<float> pos(kPos0, kPos0 + 12), pred(12);
	for (uint32_t v = 0; v < NV; ++v) {
		pred[3 * v + 0] = pos[3 * v + 0];
		pred[3 * v + 1] = pos[3 * v + 1];
		pred[3 * v + 2] = pos[3 * v + 2] + kGravity * kDt * kDt;
	}
	upload(s, pos, pred);
	for (int it = 0; it < iters; ++it) {
		if (s.step() != 0) return false;
		if (s.stepDualAttachments() != 0) return false;
	}
	std::vector<float> x;
	s.readPositions(x);
	for (float q : x) {
		if (!std::isfinite(q)) return false;
	}
	*outViol = std::sqrt(double(x[0]) * double(x[0]) + double(x[1]) * double(x[1]) +
			double(x[2]) * double(x[2]));
	return true;
}

}  // namespace

int main(int argc, char **argv) {
	setvbuf(stdout, nullptr, _IONBF, 0);
	const char *dir = (argc > 1) ? argv[1] : ".";
	cloth::AvbdSolver solver(dir);
	if (!solver.ok()) {
		std::printf("test_avbd_penalty_ramp: construction failed\n");
		return 1;
	}

	std::printf("test_avbd_penalty_ramp: attachment violation |x - fixedPos|\n");
	std::printf("  1 vertex, soft pin k=%g, gravity only; beta=0 is the\n", double(kAttachK));
	std::printf("  pre-ramp behaviour and the control arm\n\n");
	std::printf("  %6s %16s %16s %10s\n", "iters", "beta=0 (no ramp)", "beta=1e4 (ramp)",
			"ratio");

	const int iters[] = {1, 2, 4, 8, 16, 32, 64};
	const int n = int(sizeof(iters) / sizeof(iters[0]));
	double lastOff = 0.0, lastOn = 0.0;
	for (int i = 0; i < n; ++i) {
		double off = 0.0, on = 0.0;
		if (!run(solver, 0.0f, iters[i], &off) || !run(solver, 1.0e4f, iters[i], &on)) {
			std::printf("  %6d   run failed\n", iters[i]);
			return 1;
		}
		std::printf("  %6d %16.6g %16.6g %10.3g\n", iters[i], off, on,
				on > 0.0 ? off / on : INFINITY);
		lastOff = off;
		lastOn = on;
	}
	std::printf("\n");

	// The claim is that the ramp converges the constraint faster. Require
	// a full order of magnitude at 64 iterations, so a marginal or
	// noise-level difference does not count as support.
	const bool helps = lastOn < lastOff * 0.1;
	// And the arms must genuinely differ -- if beta had no effect at all
	// (a misbound uniform, a dropped write to gamma) both columns would
	// be identical and the table above would look tidy while proving
	// nothing.
	const bool differ = std::fabs(lastOn - lastOff) > 1e-12;

	if (!differ) {
		std::printf("test_avbd_penalty_ramp: ABORT -- beta=0 and beta=1e4 give\n"
					"identical results, so AVBD_BETA is not reaching the kernel and\n"
					"nothing above is evidence either way.\n");
		return 1;
	}
	if (!helps) {
		std::printf("test_avbd_penalty_ramp: FAILED -- the ramp does not converge the\n"
					"constraint faster (%.3g with, %.3g without at 64 iters). Eq. 16\n"
					"is implemented but is not earning its place.\n",
				lastOn, lastOff);
		return 1;
	}
	std::printf("test_avbd_penalty_ramp: OK -- violation %.3g with the ramp vs\n"
				"%.3g without at 64 iterations, a %.0fx improvement.\n",
			lastOn, lastOff, lastOff / lastOn);
	return 0;
}
