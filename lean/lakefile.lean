import Lake
open Lake DSL

package Cloth where

require LeanSlang from git
  "https://github.com/V-Sekai-fire/lean-slang.git" @ "v0.0.5"

-- Property testing for the AVBD specs (Cloth.Avbd.*). The existing
-- `native_decide` examples pin single fixtures; plausible quantifies
-- the same invariants over generated meshes, which is where a role or
-- offset bug would actually show up. Pinned to v4.30.0, matching this
-- project's lean-toolchain, which was bumped to v4.30.0 precisely so
-- plausible-witness-dag's ladder could be adopted below.
require plausible from git
  "https://github.com/leanprover-community/plausible" @ "v4.30.0"

@[default_target] lean_lib Cloth where

lean_exe emit_shaders where
  root := `EmitShaders

-- Iterative-deepening witness search over the plausible ladder. Shares
-- the `Level` shape (walkSteps / finBound / numInst) that witness-cpp
-- mirrors on the C++ side, so a property stated here and a property
-- stated there escalate the same way.
require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

lean_exe csr_falsify where
  root := `CsrFalsify
