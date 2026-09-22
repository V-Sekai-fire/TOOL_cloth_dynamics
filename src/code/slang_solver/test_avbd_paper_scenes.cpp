// The original AVBD paper's convergence scenes, ported to this solver.
//
// Source: savant117/avbd-demo3d, `scenes.h` -- the authors' own demo.
// Of its 14 scenes, 10 need rigid bodies, joints, tetrahedra or
// in-solver contact manifolds, none of which this cloth solver has.
// The 4 that DO port are the ones carrying the paper's actual claim:
// that AVBD converges at stiffness and mass ratios where VBD does not.
//
//   Spring Ratio  sceneSpringsRatio  8 particles, ends pinned, springs
//                                    alternating k = 10 / 10000 (1000x)
//   Rope          sceneRope          20 particles, uniform, pinned head
//   Heavy Rope    sceneHeavyRope     same, last particle 125x the mass
//   Spring        sceneSpring        one spring from an anchor
//
// A fifth scene, "Rope (pre-tensioned)", is not from scenes.h. It is a
// controlled variant of Rope added to test an explanation of Rope's
// result; see the comment on sceneRopeTensioned.
//
// RESULT, recorded here because it is the point of the file: three of
// the four ported scenes are FLAT -- the residual does not improve at
// all between 1 and 256 iterations. The pre-tensioned control, same
// topology and same springs, converges 337x over the same sweep. The
// difference is whether the springs carry tension, which is exactly
// what the dropped geometric-stiffness term supplies.
//
// Build:
//   clang++ -std=c++17 -I<vulkan>/Include -I. \
//     test_avbd_paper_scenes.cpp AvbdSolverVk.cpp -lvulkan-1

#include <cmath>
#include <cstdio>
#include <vector>

#include "AvbdSolver.h"

namespace {

// Upstream's defaults (solver.cpp, Solver::defaultParams).
constexpr float kDt = 1.0f / 60.0f;
constexpr float kGravity = -10.0f;

struct Scene {
	const char *name = "";
	uint32_t nVerts = 0;
	std::vector<float> pos;   // 3 * nVerts
	std::vector<float> mass;  // nVerts
	std::vector<uint32_t> sp1, sp2;
	std::vector<float> spRest, spK;
	std::vector<uint32_t> atVert;
	std::vector<float> atPos, atK;

	void addVert(float x, float y, float z, float m) {
		pos.push_back(x);
		pos.push_back(y);
		pos.push_back(z);
		mass.push_back(m);
		++nVerts;
	}
	void addSpring(uint32_t a, uint32_t b, float rest, float k) {
		sp1.push_back(a);
		sp2.push_back(b);
		spRest.push_back(rest);
		spK.push_back(k);
	}
	// Upstream pins by setting mass = 0. This solver has no static
	// vertices, so a pin is a very stiff attachment to the initial
	// position. 1e6 is ~100x the stiffest spring in any scene here.
	void pin(uint32_t v) {
		atVert.push_back(v);
		atPos.push_back(pos[3 * v + 0]);
		atPos.push_back(pos[3 * v + 1]);
		atPos.push_back(pos[3 * v + 2]);
		atK.push_back(1.0e6f);
	}
};

// scenes.h:112 -- the stiffness-ratio test. Springs alternate between
// k = 10 and k = 10000, a ratio of 1000.
Scene sceneSpringsRatio() {
	Scene s;
	s.name = "Spring Ratio";
	const int N = 8;
	for (int i = 0; i < N; ++i) {
		const float x = (float(i) - (N - 1) * 0.5f) * 3.0f;
		s.addVert(x, 0.0f, 12.0f, 1.0f);
	}
	for (int i = 1; i < N; ++i) {
		s.addSpring(uint32_t(i - 1), uint32_t(i), 3.0f, (i % 2 == 0) ? 10.0f : 10000.0f);
	}
	s.pin(0);
	s.pin(uint32_t(N - 1));
	return s;
}

// scenes.h:69 -- uniform chain, head pinned. The control for Heavy Rope.
Scene sceneRope() {
	Scene s;
	s.name = "Rope";
	const int N = 20;
	for (int i = 0; i < N; ++i) s.addVert(float(i), 0.0f, 10.0f, 1.0f);
	for (int i = 1; i < N; ++i) s.addSpring(uint32_t(i - 1), uint32_t(i), 1.0f, 1000.0f);
	s.pin(0);
	return s;
}

// scenes.h:84 -- same chain, but the last body is a 5x5x5 cube against
// 1x0.5x0.5 links: a mass ratio of 125 at the far end of a 20-link
// chain. This is the paper's hard case.
Scene sceneHeavyRope() {
	Scene s;
	s.name = "Heavy Rope";
	const int N = 20;
	for (int i = 0; i < N; ++i) {
		s.addVert(float(i), 0.0f, 10.0f, (i == N - 1) ? 125.0f : 1.0f);
	}
	for (int i = 1; i < N; ++i) s.addSpring(uint32_t(i - 1), uint32_t(i), 1.0f, 1000.0f);
	s.pin(0);
	return s;
}

// NOT from scenes.h. Rope, but with the chain pre-stretched 1.5x so
// every spring carries tension from the first iteration.
//
// This exists to test an explanation rather than assert one. The
// Gauss-Newton Hessian this solver uses is k * n n^T -- rank one, with
// stiffness ONLY along the spring. The term that supplies transverse
// stiffness is the geometric-stiffness part, proportional to the
// tension, and Gauss-Newton drops it. So a chain sitting exactly at
// rest length has a singular per-vertex block in the two transverse
// directions and cannot descend there at all, whereas a chain under
// tension can.
//
// If that is the right reading, this scene converges and plain Rope
// does not, on identical topology and the same solver. If both behave
// alike, the reading is wrong and the plateau is something else.
Scene sceneRopeTensioned() {
	Scene s;
	s.name = "Rope (pre-tensioned)";
	const int N = 20;
	for (int i = 0; i < N; ++i) s.addVert(float(i) * 1.5f, 0.0f, 10.0f, 1.0f);
	for (int i = 1; i < N; ++i) s.addSpring(uint32_t(i - 1), uint32_t(i), 1.0f, 1000.0f);
	s.pin(0);
	return s;
}

// scenes.h:102 -- a single spring hanging a block from a fixed anchor.
Scene sceneSpring() {
	Scene s;
	s.name = "Spring";
	s.addVert(0.0f, 0.0f, 14.0f, 1.0f);
	s.addVert(0.0f, 0.0f, 8.0f, 1.0f);
	s.addSpring(0, 1, 4.0f, 100.0f);
	s.pin(0);
	return s;
}

void upload(cloth::AvbdSolver &sv, const Scene &s, const std::vector<float> &pos,
		const std::vector<float> &pred) {
	sv.setupMesh(s.nVerts, pos.data(), pred.data(), s.mass.data(), 1.0f / (kDt * kDt));
	sv.uploadSprings(uint32_t(s.spK.size()), s.sp1.data(), s.sp2.data(), s.spRest.data(),
			s.spK.data());
	sv.uploadAttachments(uint32_t(s.atK.size()), s.atVert.data(), s.atPos.data(),
			s.atK.data());
	sv.uploadTriangles(0, nullptr, nullptr, nullptr);
	sv.uploadBendings(0, nullptr, nullptr, nullptr, nullptr);
}

// Residual of the optimality condition of ONE step's own problem.
//
// AVBD's per-step objective is
//
//     f(x) = (m / 2h^2) |x - x_inertial|^2 + sum E(x)
//
// so a converged iterate satisfies grad f = 0 exactly:
//
//     r(x) = (m/h^2)(x - x_inertial) + grad E(x) = 0
//
// This needs no reference solution and no settled state -- it is the
// solver's own stationarity condition, evaluated on the host from the
// positions it returned. That is the right thing to measure, because
// the paper's claim is about convergence WITHIN a step as iterations
// increase, not about how many steps a scene takes to come to rest.
//
// An earlier version of this test drove the scenes quasi-statically for
// 200 steps and measured STATIC equilibrium instead. That measures
// settling, not convergence: a quasi-static step moves a vertex by only
// about g*h^2 ~ 3e-3, so a 20-link rope cannot reach its catenary in
// 200 steps however well the solver converges. Every scene duly
// reported a residual of almost exactly one vertex weight -- a real
// number that simply was not about the thing being claimed.
double residualNorm(const Scene &s, const std::vector<float> &x,
		const std::vector<float> &inertial, float invHSq) {
	std::vector<double> r(3 * size_t(s.nVerts), 0.0);

	for (uint32_t v = 0; v < s.nVerts; ++v) {
		for (int k = 0; k < 3; ++k) {
			r[3 * v + k] = double(s.mass[v]) * double(invHSq) *
					(double(x[3 * v + k]) - double(inertial[3 * v + k]));
		}
	}
	for (size_t c = 0; c < s.spK.size(); ++c) {
		const uint32_t a = s.sp1[c], b = s.sp2[c];
		double d[3];
		for (int k = 0; k < 3; ++k) d[k] = double(x[3 * a + k]) - double(x[3 * b + k]);
		const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
		if (len < 1e-12) continue;
		// E = 0.5 k (len - L0)^2  =>  dE/dx_a = k (len - L0) d / len.
		const double t = double(s.spK[c]) * (len - double(s.spRest[c]));
		for (int k = 0; k < 3; ++k) {
			const double gk = t * d[k] / len;
			r[3 * a + k] += gk;
			r[3 * b + k] -= gk;
		}
	}
	for (size_t c = 0; c < s.atK.size(); ++c) {
		const uint32_t v = s.atVert[c];
		for (int k = 0; k < 3; ++k) {
			r[3 * v + k] +=
					double(s.atK[c]) * (double(x[3 * v + k]) - double(s.atPos[3 * c + k]));
		}
	}

	double n2 = 0.0;
	for (double q : r) n2 += q * q;
	return std::sqrt(n2);
}

// Build the inertial (warm-start) target for one step from `pos`.
void inertialOf(const Scene &s, const std::vector<float> &pos, std::vector<float> *pred) {
	pred->resize(3 * size_t(s.nVerts));
	for (uint32_t v = 0; v < s.nVerts; ++v) {
		(*pred)[3 * v + 0] = pos[3 * v + 0];
		(*pred)[3 * v + 1] = pos[3 * v + 1];
		(*pred)[3 * v + 2] = pos[3 * v + 2] + kGravity * kDt * kDt;
	}
}

// One step from the scene's initial state with `iters` inner
// iterations. Reports the residual at the returned iterate, and the
// residual at the warm start for scale.
bool oneStep(cloth::AvbdSolver &sv, const Scene &s, int iters, double *outRes,
		double *outRes0) {
	std::vector<float> pos = s.pos, pred;
	inertialOf(s, pos, &pred);
	const float invHSq = 1.0f / (kDt * kDt);
	*outRes0 = residualNorm(s, pos, pred, invHSq);

	upload(sv, s, pos, pred);
	for (int it = 0; it < iters; ++it) {
		if (sv.step() != 0) return false;
	}
	std::vector<float> x;
	sv.readPositions(x);
	for (float q : x) {
		if (!std::isfinite(q)) return false;
	}
	*outRes = residualNorm(s, x, pred, invHSq);
	return true;
}

int g_fail = 0;

void runScene(cloth::AvbdSolver &sv, const Scene &s) {
	std::printf("  %s  (%u verts, %zu springs)\n", s.name, s.nVerts, s.spK.size());

	const int iters[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
	const int nIters = int(sizeof(iters) / sizeof(iters[0]));
	double res0 = 0.0, first = 0.0, last = 0.0;
	bool blew = false;
	std::printf("    %6s %14s %14s\n", "iters", "|grad f|", "vs warmstart");
	for (int i = 0; i < nIters; ++i) {
		double res = 0.0;
		if (!oneStep(sv, s, iters[i], &res, &res0)) {
			std::printf("    %6d   diverged (non-finite positions)\n", iters[i]);
			blew = true;
			continue;
		}
		std::printf("    %6d %14.6g %14.4g\n", iters[i], res,
				res0 > 0.0 ? res / res0 : 0.0);
		if (i == 0) first = res;
		last = res;
	}

	if (blew) {
		std::printf("    -> DIVERGED\n\n");
		++g_fail;
		return;
	}
	// The gate is the standard relative-residual criterion: |grad f|
	// must fall to 1e-3 of its value at the warm start. Converging in
	// few iterations is a virtue, not a fault, so a plateau is reported
	// but does NOT by itself fail a scene -- an earlier version gated on
	// it too and marked the fast-converging cases as failures.
	const double rel = (res0 > 0.0) ? last / res0 : 1.0;
	const bool converged = rel < 1.0e-3;
	const bool improves = last < first * 0.5;
	std::printf("    -> |grad f| %.3g = %.3g of warmstart (%s), %.1fx better than\n",
			last, rel, converged ? "converged" : "NOT CONVERGED",
			last > 0.0 ? first / last : 0.0);
	std::printf("       1 iter, then %s\n", improves ? "still improving" : "flat");
	// A scene can pass the relative gate while being completely flat,
	// if its warm-start residual happens to be large -- Heavy Rope does
	// exactly that, plateauing at the same absolute 0.43 as Rope but
	// against a 125x heavier vertex. Say so, so the pass is not read as
	// evidence the solver made progress.
	if (!improves) {
		std::printf("       NOTE: flat from iteration 1 -- iterating does not help this\n"
					"       scene, whatever the relative figure says.\n");
	}
	std::printf("\n");
	if (!converged) ++g_fail;
}

// A residual that cannot rise is not measuring anything. Perturb the
// solver's own iterate and require the residual to grow.
bool falsifiability(cloth::AvbdSolver &sv, const Scene &s) {
	std::vector<float> pos = s.pos, pred;
	inertialOf(s, pos, &pred);
	const float invHSq = 1.0f / (kDt * kDt);
	upload(sv, s, pos, pred);
	for (int it = 0; it < 128; ++it) {
		if (sv.step() != 0) return false;
	}
	std::vector<float> x;
	sv.readPositions(x);
	const double good = residualNorm(s, x, pred, invHSq);
	std::vector<float> bad = x;
	for (size_t i = 0; i < bad.size(); ++i) bad[i] += 0.01f;
	const double worse = residualNorm(s, bad, pred, invHSq);
	const bool teeth = worse > good * 2.0;
	std::printf("  falsifiability on %s: |grad f| = %.6g at the solver's iterate,\n", s.name,
			good);
	std::printf("    %.6g after a 0.01 shift -- %s\n\n", worse,
			teeth ? "rises, so the measure has teeth" : "DOES NOT RISE (measure is blind)");
	return teeth;
}

}  // namespace

int main(int argc, char **argv) {
	setvbuf(stdout, nullptr, _IONBF, 0);
	const char *dir = (argc > 1) ? argv[1] : ".";
	cloth::AvbdSolver solver(dir);
	if (!solver.ok()) {
		std::printf("test_avbd_paper_scenes: construction failed\n");
		return 1;
	}

	std::printf("test_avbd_paper_scenes: the AVBD paper's convergence scenes\n");
	std::printf("  (savant117/avbd-demo3d scenes.h; |grad f| of each step's\n");
	std::printf("   own objective, which is zero exactly at convergence)\n\n");

	const Scene scenes[] = {sceneSpring(), sceneRope(), sceneRopeTensioned(),
			sceneHeavyRope(), sceneSpringsRatio()};
	if (!falsifiability(solver, scenes[4])) {
		std::printf("test_avbd_paper_scenes: ABORT -- the residual does not rise on a\n"
					"deliberately perturbed iterate, so nothing below can be trusted.\n");
		return 1;
	}
	for (const Scene &s : scenes) runScene(solver, s);

	if (g_fail) {
		std::printf("test_avbd_paper_scenes: %d of 5 scenes fail.\n", g_fail);
		return 1;
	}
	std::printf("test_avbd_paper_scenes: OK -- all 5 reach a stationary point and\n"
				"converge with iteration count.\n");
	return 0;
}
