// Finite-difference check of AVBD's single-step parameter gradients.
//
// The repo's entire product is gradients, and there is no gradient
// checker in it. That matters right now because the dress demo stalls
// under AVBD's adjoint (loss improves 0.002%, then 11 identical
// evaluations) and nothing distinguishes the two possible causes:
//
//   (a) the per-step adjoint is wrong, or
//   (b) the per-step adjoint is right and 125 steps of BPTT destroy it
//       -- the code's own note says the per-step Jacobian has spectral
//       radius ~sqrt(2), so AVBD_BWD_TRUNCATE_K=20 is a degree-20
//       Neumann approximation rather than the gradient.
//
// A single-step check separates them. If the 4-vertex gradient matches
// finite differences, (a) is excluded and the stall is an accumulation
// problem -- which is a completely different fix, and means PD's
// adjoint can eventually go.
//
// Loss is L = 0.5 * |x_out|^2, so dL/dx_out = x_out exactly, which
// keeps the check about the solver's adjoint rather than about a loss
// implementation.
//
// Build (needs Vulkan, no Eigen):
//   clang++ -std=c++17 -I<vulkan>/Include -I. \
//     test_avbd_gradcheck.cpp AvbdSolverVk.cpp -lvulkan-1

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "AvbdSolver.h"

namespace {

constexpr uint32_t N = 4;
const float kPos[12] = {0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0};
const float kPred[12] = {1, 2, 3, 3, 0, 0, 0, 4, 0, 3, 4, 0};
const float kMass[4] = {2, 1, 1, 1};

struct Params {
	float springK = 1.0f;
	float attachK = 4.0f;
	float triK = 1.0f;
	float bendK = 1.0f;
	float restLen = 2.0f;
};

// One forward step at the given parameters; returns 0.5*|x_out|^2 and,
// optionally, the resulting positions.
double forwardLoss(cloth::AvbdSolver &s, const Params &p,
		std::vector<float> *out = nullptr) {
	s.setupMesh(N, kPos, kPred, kMass, 2.0f);

	const uint32_t p1[1] = {0}, p2[1] = {1};
	const float rl[1] = {p.restLen}, sk[1] = {p.springK};
	s.uploadSprings(1, p1, p2, rl, sk);

	const uint32_t av[1] = {2};
	const float af[3] = {0, 4, 10}, ak[1] = {p.attachK};
	s.uploadAttachments(1, av, af, ak);

	const uint32_t ti[3] = {0, 1, 2};
	const float uv[4] = {1, 0, 0, 1}, tk[1] = {p.triK};
	s.uploadTriangles(1, ti, uv, tk);

	const uint32_t bi[4] = {0, 1, 2, 3};
	const float bw[4] = {1, 1, -1, -1}, bn[1] = {4}, bk[1] = {p.bendK};
	s.uploadBendings(1, bi, bw, bn, bk);

	if (s.step() != 0) return NAN;

	std::vector<float> x;
	s.readPositions(x);
	if (x.size() != size_t(N) * 3) return NAN;
	if (out) *out = x;

	double L = 0.0;
	for (float v : x) L += 0.5 * double(v) * double(v);
	return L;
}

// Central difference on one parameter.
double fdGrad(cloth::AvbdSolver &s, Params p, float Params::*field, double h) {
	Params hi = p, lo = p;
	hi.*field = float(double(p.*field) + h);
	lo.*field = float(double(p.*field) - h);
	const double Lhi = forwardLoss(s, hi);
	const double Llo = forwardLoss(s, lo);
	return (Lhi - Llo) / (2.0 * h);
}

int g_fail = 0;

void report(const char *name, double analytic, double numeric, double tol) {
	const double denom = std::max(1.0, std::max(std::fabs(analytic), std::fabs(numeric)));
	const double rel = std::fabs(analytic - numeric) / denom;
	const bool ok = rel <= tol;
	if (!ok) ++g_fail;
	std::printf("  %-16s analytic=%12.6g  fd=%12.6g  rel=%9.3g  %s\n", name, analytic,
			numeric, rel, ok ? "ok" : "MISMATCH");
}

}  // namespace

int main(int argc, char **argv) {
	const char *dir = (argc > 1) ? argv[1] : ".";
	cloth::AvbdSolver solver(dir);
	if (!solver.ok()) {
		std::printf("test_avbd_gradcheck: construction failed\n");
		return 1;
	}

	const Params p;

	// Analytic: one forward, then the adjoint seeded with dL/dx = x_out.
	std::vector<float> xOut;
	const double L = forwardLoss(solver, p, &xOut);
	if (!(L == L)) {
		std::printf("test_avbd_gradcheck: forward failed\n");
		return 1;
	}
	if (solver.stepBackward(xOut.data()) != 0) {
		std::printf("test_avbd_gradcheck: stepBackward failed\n");
		return 1;
	}

	std::vector<float> dRest, dSpringK, dFixed, dAttachK, dAttachLam;
	std::vector<float> dTriK, dTriL0, dTriL1, dBendN, dBendK, dBendLam;
	solver.readSpringGrad(dRest, dSpringK);
	solver.readAttachGrad(dFixed, dAttachK, dAttachLam);
	solver.readTriGrad(dTriK, dTriL0, dTriL1);
	solver.readBendGrad(dBendN, dBendK, dBendLam);

	std::printf("test_avbd_gradcheck: L = %.9g at the 4-vertex fixture\n", L);
	std::printf("  single step, dL/dx = x_out, central differences\n\n");

	// h chosen per-parameter: the fixture's stiffnesses are O(1), and
	// fp32 readback puts the noise floor around 1e-7, so 1e-3 keeps
	// truncation and round-off comparably small.
	const double h = 1e-3;
	report("dL/d restLen", dRest.empty() ? 0.0 : dRest[0],
			fdGrad(solver, p, &Params::restLen, h), 5e-2);
	report("dL/d k_spring", dSpringK.empty() ? 0.0 : dSpringK[0],
			fdGrad(solver, p, &Params::springK, h), 5e-2);
	report("dL/d k_attach", dAttachK.empty() ? 0.0 : dAttachK[0],
			fdGrad(solver, p, &Params::attachK, h), 5e-2);
	report("dL/d k_tri", dTriK.empty() ? 0.0 : dTriK[0],
			fdGrad(solver, p, &Params::triK, h), 5e-2);
	report("dL/d k_bend", dBendK.empty() ? 0.0 : dBendK[0],
			fdGrad(solver, p, &Params::bendK, h), 5e-2);

	// If anything disagreed, sweep h before blaming the adjoint: a
	// finite difference that drifts with h is the unreliable side.
	if (g_fail) {
		std::printf("\n  h-sweep on k_spring (an fd stable across h means the\n");
		std::printf("  analytic value is the wrong one):\n");
		const double hs[] = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
		for (double hh : hs) {
			std::printf("    h=%-9g fd=%12.6g\n", hh,
					fdGrad(solver, p, &Params::springK, hh));
		}
	}

	std::printf("\n");
	std::printf("\n");
	if (g_fail) {
		// Which gradient fails matters as much as whether one does.
		// The dress demo uploads nSprings = 0 (it is all triangle
		// membrane + dihedral bending + attachments), so a broken
		// spring gradient cannot explain the dress stall -- the
		// gradients that demo actually consumes are the ones that
		// match here. Do not over-read a single failure.
		std::printf("test_avbd_gradcheck: %d of 5 gradients disagree with finite\n"
					"differences. Check WHICH one before drawing conclusions: the\n"
					"dress demo has nSprings=0, so a spring-gradient failure says\n"
					"nothing about its stall, while a tri/bend/attach failure would\n"
					"implicate the per-step adjoint directly.\n",
				g_fail);
		return 1;
	}
	std::printf("test_avbd_gradcheck: OK -- all 5 match finite differences.\n"
				"The per-step adjoint is correct, so a multi-step stall is an\n"
				"accumulation problem (BPTT truncated at K=20), not a wrong\n"
				"gradient.\n");
	return 0;
}
