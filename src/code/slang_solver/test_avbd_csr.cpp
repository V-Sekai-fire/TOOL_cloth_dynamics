// CSR adjacency: unit, property, and falsifiability tests.
//
// `Cloth.Avbd.AdjacencyKwise.CsrAdj.build` and
// `Cloth.Avbd.AdjacencySpring.SpringCsr.build` are the normative
// definition of the vertex -> incident-constraint adjacency that every
// gather kernel indexes through. Lean proves its own build against
// fixtures with `native_decide` and CI Layer 0 re-checks them, but the
// C++ host build is a separate reimplementation and nothing tied the
// two together. A divergence in `role` would not crash -- it would
// silently make every gather read the wrong per-constraint slot
// ([K*c + role]) and produce plausible-but-wrong forces.
//
// Three rungs:
//
//   UNIT         the exact Lean fixtures, transcribed from
//                lean/Cloth/Avbd/AdjacencyKwise.lean  (K = 1, 3, 4)
//                lean/Cloth/Avbd/AdjacencySpring.lean (K = 2)
//
//   PROPERTY     invariants over generated meshes, via witness-cpp's
//                ladder. The round-trip property is the one that
//                matters: every (constraint, role) pair must appear
//                exactly once, at the vertex it names.
//
//   FALSIFIABILITY
//                a deliberately broken build must be CAUGHT. Without
//                this a property suite can pass because it is blind
//                rather than because the code is right -- the same
//                reason the padding test in test_avbd_vk_colored.cpp
//                is negative-controlled.
//
// Build:
//   clang++ -std=c++17 -I. -I<witness>/include -I<doctest> \
//       test_avbd_csr.cpp -o test_avbd_csr
// Needs no Eigen, no Vulkan, no GPU.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <witness/doctest.h>

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "AvbdCsr.h"

namespace {

// A generated constraint mesh: `nC` constraints of arity `k` over `nV`
// vertices, flattened as idx[k*c + r].
struct Mesh {
	uint32_t nV = 0;
	uint32_t k = 0;
	std::vector<uint32_t> idx;

	uint32_t nC() const { return k ? uint32_t(idx.size()) / k : 0u; }
};

witness::Generator<Mesh> gen_mesh() {
	return [](witness::RNG &rng, const witness::Level &lvl) {
		Mesh m;
		// Mesh size escalates with the ladder rung (walk_steps goes
		// 64 -> 512 -> 4000), kept small because CSR bugs surface at
		// tiny sizes if they surface at all. Arity 1..4 covers exactly
		// the four the AVBD kernels use: attachment, spring, triangle,
		// bending.
		const uint32_t scale = 1u + uint32_t(lvl.walk_steps) / 16u;
		m.nV = witness::gen_uint_range(1u, scale)(rng, lvl);
		m.k = witness::gen_uint_range(1u, 4u)(rng, lvl);
		const uint32_t nC = witness::gen_uint_range(0u, scale)(rng, lvl);
		witness::Generator<uint32_t> vert = witness::gen_uint_range(0u, m.nV - 1u);
		for (uint32_t i = 0; i < nC * m.k; ++i) m.idx.push_back(vert(rng, lvl));
		return m;
	};
}

// Reference for the falsifiability rung: drops the role, which is
// exactly the silent-wrong-answer failure mode described above.
void buildCsrBrokenRole(uint32_t nV, uint32_t vcap, const uint32_t *idx, uint32_t nC,
		uint32_t k, std::vector<uint32_t> &offsets, std::vector<uint32_t> &cIdx,
		std::vector<uint32_t> &role) {
	cloth::buildCsr(nV, vcap, idx, nC, k, offsets, cIdx, role);
	for (uint32_t &r : role) r = 0;
}

// Every (constraint, role) pair appears exactly once, at the vertex it
// names. This is precisely what [K*c + role] indexing in the gather
// kernels relies on.
bool roundTrips(const Mesh &m, bool broken = false) {
	std::vector<uint32_t> offsets, cIdx, role;
	(broken ? buildCsrBrokenRole : cloth::buildCsr)(
			m.nV, m.nV, m.idx.data(), m.nC(), m.k, offsets, cIdx, role);

	std::set<std::pair<uint32_t, uint32_t>> seen;
	for (uint32_t v = 0; v < m.nV; ++v) {
		for (uint32_t s = offsets[v]; s < offsets[v + 1]; ++s) {
			if (cIdx[s] >= m.nC() || role[s] >= m.k) return false;
			// The entry must name this vertex.
			if (m.idx[m.k * cIdx[s] + role[s]] != v) return false;
			if (!seen.insert({cIdx[s], role[s]}).second) return false;  // duplicate
		}
	}
	return seen.size() == size_t(m.nC()) * m.k;
}

bool offsetsWellFormed(const Mesh &m) {
	std::vector<uint32_t> offsets, cIdx, role;
	cloth::buildCsr(m.nV, m.nV, m.idx.data(), m.nC(), m.k, offsets, cIdx, role);
	if (offsets.size() != size_t(m.nV) + 1) return false;
	if (offsets[0] != 0) return false;
	for (uint32_t v = 0; v < m.nV; ++v) {
		if (offsets[v + 1] < offsets[v]) return false;  // monotone
	}
	// Total incidences == one per constraint corner.
	return offsets[m.nV] == m.nC() * m.k && cIdx.size() == size_t(m.nC()) * m.k;
}

// The Vulkan backend pads every buffer so a ragged final workgroup
// lands on a dummy slot; those padded vertices must gather an EMPTY
// range, or an overrun lane picks up a real vertex's constraints.
bool paddingGathersNothing(const Mesh &m) {
	const uint32_t vcap = m.nV + 64u;
	std::vector<uint32_t> o1, c1, r1, o2, c2, r2;
	cloth::buildCsr(m.nV, m.nV, m.idx.data(), m.nC(), m.k, o1, c1, r1);
	cloth::buildCsr(m.nV, vcap, m.idx.data(), m.nC(), m.k, o2, c2, r2);
	if (o2.size() != size_t(vcap) + 1) return false;
	for (uint32_t v = 0; v <= m.nV; ++v) {
		if (o1[v] != o2[v]) return false;
	}
	for (uint32_t v = m.nV; v <= vcap; ++v) {
		if (o2[v] != o1[m.nV]) return false;  // empty range
	}
	return c1 == c2 && r1 == r2;
}

void expectCsr(uint32_t nV, uint32_t k, const std::vector<uint32_t> &idx,
		const std::vector<uint32_t> &wantOffsets, const std::vector<uint32_t> &wantCIdx,
		const std::vector<uint32_t> &wantRole) {
	std::vector<uint32_t> offsets, cIdx, role;
	cloth::buildCsr(nV, nV, idx.data(), uint32_t(idx.size() / k), k, offsets, cIdx, role);
	CHECK(offsets == wantOffsets);
	CHECK(cIdx == wantCIdx);
	CHECK(role == wantRole);
}

}  // namespace

// ---------------------------------------------------------------------
// UNIT — the Lean fixtures, verbatim.
// ---------------------------------------------------------------------

TEST_CASE("buildCsr matches the Lean fixtures") {
	SUBCASE("K=1 attachment (AdjacencyKwise.attachFixture)") {
		// 3 verts, attachments [[0], [2]].
		expectCsr(3, 1, {0, 2}, {0, 1, 1, 2}, {0, 1}, {0, 0});
	}
	SUBCASE("K=2 spring star (AdjacencySpring.starFixture)") {
		// 3 verts, springs (0,1) (0,2) -- star centred on v0.
		expectCsr(3, 2, {0, 1, 0, 2}, {0, 2, 3, 4}, {0, 1, 0, 1}, {0, 0, 1, 1});
	}
	SUBCASE("K=2 spring chain (AdjacencySpring.chainFixture)") {
		// 4 verts, springs (0,1) (1,2) (2,3).
		expectCsr(4, 2, {0, 1, 1, 2, 2, 3}, {0, 1, 3, 5, 6}, {0, 0, 1, 1, 2, 2},
				{0, 1, 0, 1, 0, 1});
	}
	SUBCASE("K=3 triangle (AdjacencyKwise.triFixture)") {
		// 4 verts, T0 = (0,1,2), T1 = (3,2,1) -- shared edge (1,2).
		expectCsr(4, 3, {0, 1, 2, 3, 2, 1}, {0, 1, 3, 5, 6}, {0, 0, 1, 0, 1, 1},
				{0, 1, 2, 2, 1, 0});
	}
	SUBCASE("K=4 bending (AdjacencyKwise.bendFixture)") {
		// 5 verts, B0 = (0,1,2,3), B1 = (1,2,3,4).
		expectCsr(5, 4, {0, 1, 2, 3, 1, 2, 3, 4}, {0, 1, 3, 5, 7, 8},
				{0, 0, 1, 0, 1, 0, 1, 1}, {0, 1, 0, 2, 1, 3, 2, 3});
	}
}

// ---------------------------------------------------------------------
// PROPERTY — invariants over generated meshes.
// ---------------------------------------------------------------------

TEST_CASE("buildCsr invariants hold over generated meshes") {
	PROP_CHECK(Mesh, "offsets are monotone, sized nV+1, and total nC*k",
			gen_mesh(), [](const Mesh &m) { return offsetsWellFormed(m); });

	PROP_CHECK(Mesh, "every (constraint, role) appears exactly once at its vertex",
			gen_mesh(), [](const Mesh &m) { return roundTrips(m); });

	PROP_CHECK(Mesh, "padded vertices gather an empty range",
			gen_mesh(), [](const Mesh &m) { return paddingGathersNothing(m); });
}

// ---------------------------------------------------------------------
// FALSIFIABILITY — the suite must reject a knowingly-wrong build.
//
// A property that passes because it cannot see the bug is worse than
// no property. Here the round-trip check is run against a build that
// zeroes `role`; witness must report FOUND. If this ever goes green,
// the round-trip property above has lost its teeth and the gather
// kernels are unguarded.
// ---------------------------------------------------------------------

TEST_CASE("the round-trip property actually detects a broken role") {
	witness::Trial trial = witness::resolve<Mesh>(
			"role-dropping build is rejected", gen_mesh(),
			[](const Mesh &m) {
				// Meshes with k == 1 are indistinguishable (role is
				// always 0), so they are not counterexamples; skip them
				// so the generator has to find a real one.
				if (m.k <= 1 || m.nC() == 0) return true;
				return roundTrips(m, /*broken=*/true);
			});
	INFO(trial.message);
	CHECK(trial.outcome == witness::Outcome::FOUND);
}
