import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherSpringBackward` — adjoint of vbd_gather_spring (PR-G continued)

Backward kernel for `vbd_gather_spring`.

!! THE HESSIAN PATH OF THIS KERNEL IS WRONG. !!  The gradient path is
correct; the Hessian path is not, and is not yet fixed. See
`Cloth.Avbd.SpringHessAdjoint`, which states the adjoint identity this
kernel fails.

The gradient path. The forward gathers per-spring gradients into
per-vertex scratch with a sign flip on endpoint b, so the adjoint
dispatches one thread per spring and fans both endpoint cotangents back
out with the matching signs:

  per c:
    p1 = springP1Idx[c],  p2 = springP2Idx[c]
    v_springGradA[c] = v_g[p1] - v_g[p2]       -- +1*p1  +  (-1)*p2

That much is right.

The Hessian path. The doc that stood here described a forward in which
each spring holds a SINGLE SCALAR, broadcast onto the three diagonal
entries of both endpoints, making the adjoint a sum of two traces.
No such forward exists in this repo, and never did. `SpringForce.lean`
emits

    hess[6c .. 6c+5] = k * (d (x) d) / len^2

a rank-1 symmetric 3x3 stored as six components, and
`VbdGatherSpring.lean` adds all six into both endpoints:

    hScratch[6v + j] += springHess[6c + j]      j = 0..5

(The same object appears as `k * n n^T` in the authors' own reference
implementation, savant117/avbd-demo3d, `Spring::updatePrimal`.)

The forward is linear in `springHess`, so its adjoint is forced -- there
is exactly one correct answer, the componentwise transpose:

    v_springHess[6c + j] = v_H[6*p1 + j] + v_H[6*p2 + j]

What ships instead is `v_springHess[c] = trace(v_H[p1]) + trace(v_H[p2])`,
which is wrong twice over: the buffer is the wrong SHAPE (length
N_springs where 6*N_springs is needed) and, because `n n^T` is rank one
with generally non-zero off-diagonals, discarding the off-diagonal
cotangents throws away most of the block rather than a small correction.

Consequence: d L / d k_spring is wrong (measured 8.43x off against
finite differences in `test_avbd_gradcheck`). Springs are the only
constraint family affected -- attachment, membrane and bending all
check out -- and the dress demo uploads nSprings = 0, so this does not
explain the dress. Fixing it changes this kernel's binding 5 from
`float` of length N_springs to `float` of length 6*N_springs, and both
backends' `readSpringGrad` with it.

Bindings (set 0):

  0  StructuredBuffer<uint>     springP1Idx       length = N_springs
  1  StructuredBuffer<uint>     springP2Idx       length = N_springs
  2  StructuredBuffer<float3>   v_g               length = N_verts (cotangent)
  3  StructuredBuffer<float>    v_H               length = 6·N_verts (cotangent, sym 3x3)
  4  RWStructuredBuffer<float3> v_springGradA     length = N_springs (output)
  5  RWStructuredBuffer<float>  v_springHess      length = N_springs (output)
-/

namespace Cloth.SlangCodegen.VbdGatherSpringBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"      (.member (.var "tid") "x")
  , .declInit u  "p1"     (.index (.var "springP1Idx") (.var "c"))
  , .declInit u  "p2"     (.index (.var "springP2Idx") (.var "c"))
  , .declInit u  "hb1"    (.bin "*" (.litUint 6) (.var "p1"))
  , .declInit u  "hb2"    (.bin "*" (.litUint 6) (.var "p2"))
  , .declInit f3 "vg1"    (.index (.var "v_g") (.var "p1"))
  , .declInit f3 "vg2"    (.index (.var "v_g") (.var "p2"))
  , .declInit f  "vH1xx"  (.index (.var "v_H") (.var "hb1"))
  , .declInit f  "vH1yy"  (.index (.var "v_H") (.bin "+" (.var "hb1") (.litUint 3)))
  , .declInit f  "vH1zz"  (.index (.var "v_H") (.bin "+" (.var "hb1") (.litUint 5)))
  , .declInit f  "vH2xx"  (.index (.var "v_H") (.var "hb2"))
  , .declInit f  "vH2yy"  (.index (.var "v_H") (.bin "+" (.var "hb2") (.litUint 3)))
  , .declInit f  "vH2zz"  (.index (.var "v_H") (.bin "+" (.var "hb2") (.litUint 5)))
  , .assign (.index (.var "v_springGradA") (.var "c"))
      (.call "float3"
        [ .bin "-" (.member (.var "vg1") "x") (.member (.var "vg2") "x")
        , .bin "-" (.member (.var "vg1") "y") (.member (.var "vg2") "y")
        , .bin "-" (.member (.var "vg1") "z") (.member (.var "vg2") "z") ])
  , .assign (.index (.var "v_springHess") (.var "c"))
      (.bin "+"
        (.bin "+" (.bin "+" (.var "vH1xx") (.var "vH1yy")) (.var "vH1zz"))
        (.bin "+" (.bin "+" (.var "vH2xx") (.var "vH2yy")) (.var "vH2zz")))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "springP1Idx"   (.roBuf u)
      , bnd 1 "springP2Idx"   (.roBuf u)
      , bnd 2 "v_g"           (.roBuf f3)
      , bnd 3 "v_H"           (.roBuf f)
      , bnd 4 "v_springGradA" (.rwBuf f3)
      , bnd 5 "v_springHess"  (.rwBuf f)
      ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [{ name := "tid", type := .vec .uint 3
                 , semantic := Semantic.svDispatchThreadId
                 , binding := none, space := none }]
      body   := body
    }] }

def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<uint> springP1Idx;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> springP2Idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(3, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> v_springGradA;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> v_springHess;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint p1 = springP1Idx[c];
  uint p2 = springP2Idx[c];
  uint hb1 = (6u * p1);
  uint hb2 = (6u * p2);
  float3 vg1 = v_g[p1];
  float3 vg2 = v_g[p2];
  float vH1xx = v_H[hb1];
  float vH1yy = v_H[(hb1 + 3u)];
  float vH1zz = v_H[(hb1 + 5u)];
  float vH2xx = v_H[hb2];
  float vH2yy = v_H[(hb2 + 3u)];
  float vH2zz = v_H[(hb2 + 5u)];
  v_springGradA[c] = float3((vg1.x - vg2.x), (vg1.y - vg2.y), (vg1.z - vg2.z));
  v_springHess[c] = (((vH1xx + vH1yy) + vH1zz) + ((vH2xx + vH2yy) + vH2zz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherSpringBackward
