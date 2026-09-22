#ifndef OMEGAENGINE_AVBDASSEMBLY_H
#define OMEGAENGINE_AVBDASSEMBLY_H

#include <cstdint>
#include <vector>

// The AVBD analogue of `Constraint::addConstraint`.
//
// `addConstraint` lets each constraint contribute its own rows to a
// sparse system without the caller knowing the concrete type. AVBD needs
// the same protocol for a different destination: the per-family arrays
// that `AvbdSolver::upload*` consumes.
//
// Before this, `Simulation::initializePrefactoredMatrices` hand-extracted
// each family by walking the typed containers directly -- `springs`,
// `mesh`, `bendingConstraints`, `sysMat[i].attachments` -- while the
// authoritative constraint list for that sysMat lived in
// `sysMat[i].constraints`. Two descriptions of the same set, agreeing
// only because `createConstraints` happens to build the second from the
// first. A fifth constraint type, or a sysMat whose constraint list was
// filtered, would silently never reach the solver.
//
// Each constraint appends itself to the family it belongs to. Ordering
// within a family follows the order of `sysMat[i].constraints`, which is
// the order `createConstraints` pushed them: springs, triangles,
// bendings, attachments.
struct AvbdAssembly {
	// Springs: endpoint pair, rest length, stiffness.
	std::vector<uint32_t> spP1, spP2;
	std::vector<float> spRest, spK;

	// Attachments: pinned vertex, world anchor (3 per), stiffness.
	std::vector<uint32_t> atVert;
	std::vector<float> atFixed;
	std::vector<float> atK;

	// Triangles: vertex triple, row-major 2x2 inv_deltaUV (4 per),
	// stiffness.
	std::vector<uint32_t> triIdx;
	std::vector<float> triInvUV;
	std::vector<float> triK;

	// Bending: 4-vertex stencil, cotangent weights (4 per), rest
	// magnitude, stiffness.
	std::vector<uint32_t> bendIdx;
	std::vector<float> bendWeight;
	std::vector<float> bendNTarget;
	std::vector<float> bendK;

	// AVBD_RAW_STIFFNESS=1. PD weights membrane energy by
	// k_stiff*area_rest and bending by k_stiff*3/(A0+A1), while the Slang
	// kernels consume `stiffness[c]` raw. Uploading bare k_stiff made
	// membrane ~57x too stiff and bending ~85x too soft on the dress
	// mesh. The constraints therefore contribute constrainWeightSqrt^2
	// (PD's effective weight) unless this is set, which restores the old
	// behaviour for A/B comparison.
	bool rawStiffness = false;

	// AVBD_NO_MEMBRANE=1 / AVBD_NO_BENDING=1. Kept here rather than at
	// the upload site so a constraint's decision not to participate lives
	// with the constraint, not with the caller.
	bool membrane = true;
	bool bending = true;

	uint32_t nSprings() const { return uint32_t(spK.size()); }
	uint32_t nAttachments() const { return uint32_t(atK.size()); }
	uint32_t nTriangles() const { return uint32_t(triK.size()); }
	uint32_t nBendings() const { return uint32_t(bendK.size()); }
};

#endif // OMEGAENGINE_AVBDASSEMBLY_H
