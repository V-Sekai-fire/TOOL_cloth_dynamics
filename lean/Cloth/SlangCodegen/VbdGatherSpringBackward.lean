import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherSpringBackward` — adjoint of vbd_gather_spring (PR-G continued)

Backward kernel for `vbd_gather_spring`.

The gradient path gathers each spring's gradient into both endpoints
with a sign flip on endpoint b, so the adjoint dispatches one thread per
spring and fans both endpoint cotangents back out with matching signs:

  per c:
    p1 = springP1Idx[c],  p2 = springP2Idx[c]
    v_springGradA[c] = v_g[p1] - v_g[p2]       -- +1*p1  +  (-1)*p2

The Hessian path is the componentwise transpose. `SpringForce.lean`
emits

    hess[6c .. 6c+5] = k * (d (x) d) / len^2

a rank-1 symmetric 3x3 stored as six components (the same object the
authors' own reference, savant117/avbd-demo3d `Spring::updatePrimal`,
writes as `k * n n^T`), and `VbdGatherSpring.lean` adds all six into
both endpoints:

    hScratch[6v + j] += springHess[6c + j]      j = 0..5

That is linear in `springHess`, so its adjoint is forced -- there is
exactly one correct answer:

    v_springHess[6c + j] = v_H[6*p1 + j] + v_H[6*p2 + j]

WHAT THIS KERNEL USED TO DO, and why it is worth recording: it computed
one scalar per spring,

    v_springHess[c] = trace(v_H[p1]) + trace(v_H[p2])

and its doc block described a forward in which each spring holds a
single scalar broadcast onto three diagonal entries. No such forward
existed in this repo. The result was wrong twice over -- the buffer was
the wrong SHAPE (length N_springs where 6*N_springs is needed) and,
because `n n^T` is rank one with generally non-zero off-diagonals,
discarding the off-diagonal cotangents threw away most of the block.
`Cloth.Avbd.SpringHessAdjoint` states the adjoint identity
<A x, y> = <x, A* y> that the trace form fails and this form satisfies,
and proves that even restricted to the diagonal-only case the old doc
assumed was general, the trace form was off by exactly 3.

Symptom before the fix: d L / d k_spring disagreed with finite
differences by 8.43x in `test_avbd_gradcheck`, springs alone among the
four constraint families.

Bindings (set 0):

  0  StructuredBuffer<uint>     springP1Idx       length = N_springs
  1  StructuredBuffer<uint>     springP2Idx       length = N_springs
  2  StructuredBuffer<float3>   v_g               length = N_verts (cotangent)
  3  StructuredBuffer<float>    v_H               length = 6·N_verts (cotangent, sym 3x3)
  4  RWStructuredBuffer<float3> v_springGradA     length = N_springs (output)
  5  RWStructuredBuffer<float>  v_springHess      length = 6 * N_springs (output)
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
  , .declInit u  "hbc"    (.bin "*" (.litUint 6) (.var "c"))
  , .declInit f3 "vg1"    (.index (.var "v_g") (.var "p1"))
  , .declInit f3 "vg2"    (.index (.var "v_g") (.var "p2"))
  , .assign (.index (.var "v_springGradA") (.var "c"))
      (.call "float3"
        [ .bin "-" (.member (.var "vg1") "x") (.member (.var "vg2") "x")
        , .bin "-" (.member (.var "vg1") "y") (.member (.var "vg2") "y")
        , .bin "-" (.member (.var "vg1") "z") (.member (.var "vg2") "z") ])
  -- The forward adds all six components of the spring's block into
  -- both endpoints, so the adjoint is the componentwise transpose.
  , .assign (.index (.var "v_springHess") (.var "hbc"))
      (.bin "+" (.index (.var "v_H") (.var "hb1"))
                (.index (.var "v_H") (.var "hb2")))
  ] ++ (List.range 5).map (fun j =>
    let o : SlangExpr := .litUint (j + 1)
    .assign (.index (.var "v_springHess") (.bin "+" (.var "hbc") o))
      (.bin "+" (.index (.var "v_H") (.bin "+" (.var "hb1") o))
                (.index (.var "v_H") (.bin "+" (.var "hb2") o))))

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
  uint hbc = (6u * c);
  float3 vg1 = v_g[p1];
  float3 vg2 = v_g[p2];
  v_springGradA[c] = float3((vg1.x - vg2.x), (vg1.y - vg2.y), (vg1.z - vg2.z));
  v_springHess[hbc] = (v_H[hb1] + v_H[hb2]);
  v_springHess[(hbc + 1u)] = (v_H[(hb1 + 1u)] + v_H[(hb2 + 1u)]);
  v_springHess[(hbc + 2u)] = (v_H[(hb1 + 2u)] + v_H[(hb2 + 2u)]);
  v_springHess[(hbc + 3u)] = (v_H[(hb1 + 3u)] + v_H[(hb2 + 3u)]);
  v_springHess[(hbc + 4u)] = (v_H[(hb1 + 4u)] + v_H[(hb2 + 4u)]);
  v_springHess[(hbc + 5u)] = (v_H[(hb1 + 5u)] + v_H[(hb2 + 5u)]);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherSpringBackward
