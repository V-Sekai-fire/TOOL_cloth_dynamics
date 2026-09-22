import Plausible


/-!
# `Cloth.Avbd.SpringHessAdjoint` — the Hessian path of the spring gather

`vbd_gather_spring` gathers each spring's Hessian block into the
per-vertex scratch of *both* its endpoints, six components at a time:

```
for v, for (c, r) in vertSpring-CSR[v]:
    hScratch[6*v + j] += springHess[6*c + j]        j = 0..5
```

It is linear in `springHess`, so its adjoint is forced: there is exactly
one correct answer and it is the transpose.

`vbd_gather_spring_backward` DID NOT compute that transpose. Its doc
block described a *different* forward -- one where each spring holds a
single scalar broadcast onto three diagonal entries:

```
H_v_diag += springHessScalar[c]
v_springHess[c] = trace(v_H[p1]) + trace(v_H[p2])
```

No such forward ever existed in this repo. `VbdGatherSpring.lean`
stores six independent components per spring, so `v_springHess` was the
wrong *shape* (length `N_springs`, needs `6*N_springs`) and the trace
was the wrong *value*; off-diagonal cotangents were discarded outright.
The downstream `spring_force_backward` compounded it, reading that
scalar and omitting the `d(hess)/d(d)` path from `v_p_d` entirely.

Both kernels have since been corrected, and this file is what pinned
down what "correct" meant before either was touched. It is kept as a
regression: if someone reintroduces a trace, or reshapes the buffer,
the sweeps below fail.

The discriminating statement is the adjoint identity: for a linear map
`A`, an adjoint `A*` must satisfy

    ⟨A x, y⟩ = ⟨x, A* y⟩       for all x, y.

That is checked below over generated meshes and cotangents. It holds
for the componentwise transpose and fails for the trace form, which is
the whole claim.

Arithmetic is `Int`, not `Float`: the identity is exact, so a rounding
tolerance would only blunt the test.
-/

namespace Cloth.Avbd.SpringHessAdjoint

/-- A spring's two endpoint vertex ids. -/
structure Spring where
  p1 : Nat
  p2 : Nat
  deriving Repr, Inhabited

/-- Six components of a symmetric 3x3, in the kernel's storage order
    `xx, xy, xz, yy, yz, zz`. -/
abbrev Sym3 := Array Int

def zero6 : Sym3 := #[0, 0, 0, 0, 0, 0]

def get6 (a : Array Int) (base i : Nat) : Int :=
  a.getD (base + i) 0

def add6 (a b : Sym3) : Sym3 :=
  (Array.range 6).map (fun i => a.getD i 0 + b.getD i 0)

def dot6 (a b : Sym3) : Int :=
  (Array.range 6).foldl (fun acc i => acc + a.getD i 0 * b.getD i 0) 0

/-- Trace of a symmetric block in the kernel's storage order: the
    entries at 0, 3 and 5. -/
def trace6 (a : Sym3) : Int :=
  a.getD 0 0 + a.getD 3 0 + a.getD 5 0

/-- FORWARD, exactly as `vbd_gather_spring` computes it: every spring
    adds all six of its components to both endpoints' scratch. -/
def gatherHess (nV : Nat) (springs : Array Spring) (springHess : Array Sym3) :
    Array Sym3 :=
  springs.zipIdx.foldl
    (fun acc (s, c) =>
      let h := springHess.getD c zero6
      let acc := if s.p1 < nV then acc.set! s.p1 (add6 (acc.getD s.p1 zero6) h) else acc
      if s.p2 < nV then acc.set! s.p2 (add6 (acc.getD s.p2 zero6) h) else acc)
    (Array.replicate nV zero6)

/-- ADJOINT, the componentwise transpose. Each spring collects the
    cotangent of both endpoints, component by component. -/
def gatherHessAdjoint (nV : Nat) (springs : Array Spring) (vH : Array Sym3) :
    Array Sym3 :=
  springs.map (fun s =>
    let a := if s.p1 < nV then vH.getD s.p1 zero6 else zero6
    let b := if s.p2 < nV then vH.getD s.p2 zero6 else zero6
    add6 a b)

/-- ADJOINT AS SHIPPED in `vbd_gather_spring_backward`: one scalar per
    spring, the sum of both endpoints' traces. -/
def gatherHessAdjointTrace (nV : Nat) (springs : Array Spring) (vH : Array Sym3) :
    Array Int :=
  springs.map (fun s =>
    let a := if s.p1 < nV then trace6 (vH.getD s.p1 zero6) else 0
    let b := if s.p2 < nV then trace6 (vH.getD s.p2 zero6) else 0
    a + b)

/-- ⟨A x, y⟩ over the vertex side. -/
def pairingVerts (out vH : Array Sym3) : Int :=
  (Array.range out.size).foldl
    (fun acc v => acc + dot6 (out.getD v zero6) (vH.getD v zero6)) 0

/-- ⟨x, A* y⟩ over the spring side. -/
def pairingSprings (springHess adj : Array Sym3) : Int :=
  (Array.range springHess.size).foldl
    (fun acc c => acc + dot6 (springHess.getD c zero6) (adj.getD c zero6)) 0

/-- ⟨x, A* y⟩ under the shipped scalar adjoint. A scalar per spring can
    only pair against a scalar, and the only scalar the forward offers
    is the trace of that spring's block -- which is the generous
    reading, since the alternative is that the pairing is undefined. -/
def pairingSpringsTrace (springHess : Array Sym3) (adj : Array Int) : Int :=
  (Array.range springHess.size).foldl
    (fun acc c => acc + trace6 (springHess.getD c zero6) * adj.getD c 0) 0

/-! ## The adjoint identity

`⟨A x, y⟩ = ⟨x, A* y⟩`. For a linear `A` this holds for the true
adjoint and for nothing else, so it is a complete test, not a heuristic.
-/

/-- Holds iff the componentwise adjoint really is the transpose. -/
def identityHolds (nV : Nat) (springs : Array Spring)
    (springHess vH : Array Sym3) : Bool :=
  pairingVerts (gatherHess nV springs springHess) vH
    == pairingSprings springHess (gatherHessAdjoint nV springs vH)

/-- Same identity for the shipped trace form. -/
def identityHoldsTrace (nV : Nat) (springs : Array Spring)
    (springHess vH : Array Sym3) : Bool :=
  pairingVerts (gatherHess nV springs springHess) vH
    == pairingSpringsTrace springHess (gatherHessAdjointTrace nV springs vH)

/-! ### A deterministic family of meshes and cotangents

Cheap LCG-derived data, so the sweep below is exhaustive over a stated
range rather than a handful of hand-picked cases. -/

def rnd (seed i : Nat) : Int :=
  (Int.ofNat ((seed * 1103515245 + i * 12345 + 7) % 17)) - 8

def mkSym (seed i : Nat) : Sym3 :=
  (Array.range 6).map (fun j => rnd seed (i * 6 + j))

def mkSprings (seed nV nS : Nat) : Array Spring :=
  (Array.range nS).map (fun c =>
    { p1 := (seed + 3 * c) % nV
    , p2 := (seed + 3 * c + 1 + (c % (nV - 1))) % nV })

/-- One trial: build a mesh from `seed` and report both verdicts. -/
def trial (seed : Nat) : Bool × Bool :=
  let nV := 2 + seed % 5
  let nS := 1 + seed % 4
  let springs := mkSprings seed nV nS
  let sh := (Array.range nS).map (fun c => mkSym seed c)
  let vH := (Array.range nV).map (fun v => mkSym (seed + 101) v)
  (identityHolds nV springs sh vH, identityHoldsTrace nV springs sh vH)

/-- The componentwise adjoint satisfies the identity on every trial. -/
example : (List.range 400).all (fun s => (trial s).1) = true := by native_decide

/-- FALSIFIABILITY. The shipped trace form must FAIL somewhere, or this
    whole file is checking a tautology and proves nothing about the
    kernel. We require an outright majority of counterexamples, not
    merely one, so the result cannot hinge on a lucky seed. -/
example : ((List.range 400).filter (fun s => !(trial s).2)).length > 200 := by
  native_decide

/-! ### Even the most favourable case is off by a constant factor

The retracted doc assumed each spring carried one scalar broadcast onto
three diagonal entries. Restrict to exactly that case -- every block a
multiple of the identity, all off-diagonals zero -- and the trace form
STILL does not satisfy the identity. It overcounts by exactly 3, since
`trace` sums three diagonal entries on the spring side and again on the
vertex side where one factor of the pairing already supplied them.

That matters for reading `test_avbd_gradcheck`'s 8.43x error on
d L / d k_spring: a plain factor of 3 would be the whole story only if
the off-diagonals vanished, and for `k n n^T` they do not. So the
measured error is the 3x overcount and the discarded off-diagonals
together, and fixing only one of the two will not close it.
-/

def diagPair (seed : Nat) : Int × Int :=
  let nV := 2 + seed % 5
  let nS := 1 + seed % 4
  let springs := mkSprings seed nV nS
  let sh := (Array.range nS).map (fun c => let t := rnd seed c; #[t, 0, 0, t, 0, t])
  let vH := (Array.range nV).map (fun v => let t := rnd (seed + 101) v; #[t, 0, 0, t, 0, t])
  ( pairingVerts (gatherHess nV springs sh) vH
  , pairingSpringsTrace sh (gatherHessAdjointTrace nV springs vH) )

/-- The identity fails in the diagonal case too... -/
example : (List.range 200).all (fun s => let (a, b) := diagPair s; a == b) = false := by
  native_decide

/-- ...and it fails by exactly a factor of 3, every time. -/
example : (List.range 200).all (fun s => let (a, b) := diagPair s; b == 3 * a) = true := by
  native_decide

/-- Off the diagonal case, no constant factor rescues it: the trace form
    is not a rescaling of the truth, it is a different map. -/
example :
    ((List.range 400).filter (fun s =>
      let nV := 2 + s % 5
      let nS := 1 + s % 4
      let springs := mkSprings s nV nS
      let sh := (Array.range nS).map (fun c => mkSym s c)
      let vH := (Array.range nV).map (fun v => mkSym (s + 101) v)
      let a := pairingVerts (gatherHess nV springs sh) vH
      let b := pairingSpringsTrace sh (gatherHessAdjointTrace nV springs vH)
      b != 3 * a)).length > 200 := by
  native_decide

end Cloth.Avbd.SpringHessAdjoint
