import Cloth.Avbd.Coloring
import Plausible
import PlausibleWitnessDag

/-!
# `Cloth.Avbd.ColoringProperties` — property and falsifiability tests for the coloring

`Coloring` pins `assignColors` with `native_decide` on three fixtures
(disconnected, two triangles sharing an edge, pentagon cycle). Those
prove the coloring is right *there*.

The coloring carries the strongest safety obligation in the AVBD port:
the vertex-block-update kernel dispatches one color at a time and
relies on same-colored vertices sharing no constraint. If that ever
fails, two threads write overlapping constraint state and the result is
a race — which on a GPU means nondeterministic, plausible-looking
cloth rather than a crash. That is precisely the failure a fixture
cannot rule out, so it is stated here as a property instead.

Rungs, matching `Cloth.Avbd.CsrProperties`:

* **Property** (`plausible`) — the separation and partition invariants
  over generated constraint graphs.
* **Falsifiability** (`plausible-witness-dag`) — a coloring that
  ignores one adjacency must be *caught*. Without this the separation
  property could be passing because it is blind.
-/

namespace Cloth.Avbd.ColoringProperties

open Cloth.Avbd.Coloring

/-- Generated constraints may name vertices outside `[0, nVerts)`. -/
def clampMesh (nVerts : Nat) (cs : List (List Nat)) : List (List Nat) :=
  if nVerts = 0 then [] else cs.map (fun inc => inc.map (fun v => v % nVerts))

/-- Adjacency as the spec defines it: `u ~ v` iff some constraint lists
both. Recomputed here independently of `Coloring.buildAdj` so the
property checks the *definition*, not the implementation's own notion
of adjacency. -/
def adjacent (m : List (List Nat)) (u v : Nat) : Bool :=
  u != v && m.any (fun inc => inc.contains u && inc.contains v)

/-- **The safety property.** No two adjacent vertices share a color.
This is the invariant the kernel's race-freedom rests on. -/
def separationHolds (nVerts : Nat) (cs : List (List Nat)) : Bool :=
  let m := clampMesh nVerts cs
  let a := assignColors nVerts m
  (List.range nVerts).all fun u =>
    (List.range nVerts).all fun v =>
      !(adjacent m u v) || a.vertColor[u]! != a.vertColor[v]!

/-- `vertPerm` is a permutation of `[0, nVerts)` — every vertex is
dispatched exactly once, so none is skipped and none updated twice. -/
def permIsBijection (nVerts : Nat) (cs : List (List Nat)) : Bool :=
  let a := assignColors nVerts (clampMesh nVerts cs)
  a.vertPerm.size == nVerts
    && (List.range nVerts).all (fun v => a.vertPerm.toList.contains v)

/-- `colorOffsets` partitions `vertPerm`: it is monotone, starts at 0,
ends at `nVerts`, and has one entry per color plus a sentinel. -/
def offsetsPartition (nVerts : Nat) (cs : List (List Nat)) : Bool :=
  let a := assignColors nVerts (clampMesh nVerts cs)
  let o := a.colorOffsets
  o.size == a.numColors + 1
    && o[0]! == 0
    && o[a.numColors]! == nVerts
    && (List.range a.numColors).all (fun k => o[k]! ≤ o[k + 1]!)

/-- Every vertex in color `k`'s slice really has color `k`. Ties the
permutation back to `vertColor`. -/
def sliceMatchesColor (nVerts : Nat) (cs : List (List Nat)) : Bool :=
  let a := assignColors nVerts (clampMesh nVerts cs)
  (List.range a.numColors).all fun k =>
    (List.range (a.colorOffsets[k + 1]! - a.colorOffsets[k]!)).all fun j =>
      let v := a.vertPerm[a.colorOffsets[k]! + j]!
      a.vertColor[v]! == k

/-! ## Property rung -/

-- Kept small: a coloring bug needs only a handful of vertices to show
-- up, and `separationHolds` is quadratic in nVerts.
#test ∀ (nVerts : Nat) (cs : List (List Nat)),
  separationHolds (nVerts % 6) cs = true

#test ∀ (nVerts : Nat) (cs : List (List Nat)),
  permIsBijection (nVerts % 6) cs = true

#test ∀ (nVerts : Nat) (cs : List (List Nat)),
  offsetsPartition (nVerts % 6) cs = true

#test ∀ (nVerts : Nat) (cs : List (List Nat)),
  sliceMatchesColor (nVerts % 6) cs = true

/-! ## Falsifiability rung

A coloring that colors every vertex 0 is maximally broken: it is a
valid *partition* but violates separation on any mesh with an edge.
`separationHolds` must reject it. -/

/-- Everything one color — the degenerate "coloring" that would let
every vertex dispatch in a single race-prone batch. -/
def assignColorsAllZero (nVerts : Nat) (_cs : List (List Nat)) : ColorAssignment :=
  { vertColor := Array.replicate nVerts 0
  , colorOffsets := #[0, nVerts]
  , vertPerm := (List.range nVerts).toArray
  , numColors := 1 }

/-- `separationHolds`, against the all-zero coloring. `false` means the
broken coloring was caught on this mesh. -/
def brokenSeparationHolds (nVerts : Nat) (cs : List (List Nat)) : Bool :=
  let m := clampMesh nVerts cs
  let a := assignColorsAllZero nVerts m
  (List.range nVerts).all fun u =>
    (List.range nVerts).all fun v =>
      !(adjacent m u v) || a.vertColor[u]! != a.vertColor[v]!

/-- Decode a candidate into a small constraint graph. Deterministic and
total, so the search is reproducible. Arity ≥ 2 and ≥ 2 vertices, since
a graph with no edge has no adjacency to violate and is a degenerate
case rather than a counterexample. -/
def decodeGraph (n : Nat) : Nat × List (List Nat) :=
  let nVerts := 2 + n % 3        -- 2..4 vertices
  let k := 2 + (n / 3) % 3       -- arity 2..4
  let nC := 1 + (n / 9) % 2      -- 1..2 constraints
  let seed := n / 18
  let mesh : List (List Nat) :=
    (List.range nC).map fun c =>
      (List.range k).map fun r => (seed + 3 * c + 5 * r) % nVerts
  (nVerts, mesh)

/-- A candidate is a witness when the all-zero coloring FAILS
separation there — i.e. the property detected the bug. -/
def brokenColoringIsCaught (_lvl : PlausibleWitnessDag.Level) (n : Nat) : Bool :=
  let (nVerts, mesh) := decodeGraph n
  -- Need at least one genuine adjacency, else there is nothing to catch.
  let hasEdge := (List.range nVerts).any fun u =>
    (List.range nVerts).any fun v => adjacent mesh u v
  hasEdge && !(brokenSeparationHolds nVerts mesh)

/-- Deterministic readback: first candidate below the bound that
catches the all-zero coloring. -/
def readbackGraph (steps : Nat) : PlausibleWitnessDag.Readback (Nat × List (List Nat)) :=
  let found := (List.range steps).find? fun n =>
    brokenColoringIsCaught
      { idx := 0, walkSteps := steps, finBound := steps, numInst := 1 } n
  match found with
  | some n => { value := decodeGraph n, found := true, witnessIdx := n,
                budgetHit := false }
  | none => { value := (0, []), found := false, witnessIdx := 0,
              budgetHit := true }

/-- Run the falsifiability search. A `provablyNone` here would mean
`separationHolds` cannot see a maximally broken coloring, and the
kernel's race-freedom argument would be resting on nothing. -/
def checkFalsifiability : IO Unit := do
  let (graph, idx, trace) ← PlausibleWitnessDag.resolve
    "all-one-color coloring is caught by separationHolds"
    brokenColoringIsCaught readbackGraph
  IO.println s!"[coloring-falsifiability] witness idx={idx} graph={graph}"
  IO.println s!"[coloring-falsifiability] {repr trace}"

end Cloth.Avbd.ColoringProperties
