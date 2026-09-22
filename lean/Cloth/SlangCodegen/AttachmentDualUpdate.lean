import LeanSlang

/-!
# `Cloth.SlangCodegen.AttachmentDualUpdate` — AVBD augmented-Lagrangian dual ramp

First kernel for PR-F augmented Lagrangian (todo.md). The convergence
probe in #73 showed AVBD's per-vertex Gauss-Seidel oscillates at high
constraint stiffness — at least one attachment vertex stays
`conv_max ≈ 0.7` even at 64 iters. The Roblox SIGGRAPH 2025 AVBD paper
addresses this exact failure mode with augmented Lagrangian dual
updates: progressively ramp a Lagrange multiplier λ_c per hard
constraint to drive the constraint violation toward zero.

For each attachment `c` pinning vertex `v = vertIdx[c]` to anchor
`fixedPos[c]` with augmented-Lagrangian penalty `gamma[c]`:

```
C(x_v)  = x_v − fixedPos[c]              -- constraint violation (3-vec)
λ_c    ← λ_c + gamma[c] · C(x_v)         -- dual ascent
```

`λ` becomes an external force the attachment force kernel reads each
outer iter. The combined effect on the AVBD vertex-block update is:

```
gradV[c]  = (k + γ_c) · C(x_v) + λ_c     -- when attachment_force_al
                                            (next PR) adds the dual term
hess      = (k + γ_c)
```

This kernel runs once per AVBD outer-iter sweep, AFTER vbd_solve_apply
has updated positions. The accumulated λ is then read by the
attachment-force kernel on the next iter's force compute.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions    length = N_verts
  1  StructuredBuffer<uint>     vertIdx      length = N_attach
  2  StructuredBuffer<float3>   fixedPos     length = N_attach
  3  StructuredBuffer<float>    gamma        length = N_attach
  4  RWStructuredBuffer<float3> lambda       length = N_attach
                                              (initialized to zero
                                              at scene init; accumulates
                                              across iters / steps)
-/

namespace Cloth.SlangCodegen.AttachmentDualUpdate

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"   (.member (.var "tid") "x")
  , .ifThen
      (.bin ">=" (.var "c") (.member (.var "params") "count"))
      [ .ret none ]
      []
  , .declInit u  "v"   (.index (.var "vertIdx") (.var "c"))
  , .declInit f3 "p"   (.index (.var "positions") (.var "v"))
  , .declInit f3 "fp"  (.index (.var "fixedPos") (.var "c"))
  , .declInit f  "g"   (.index (.var "gamma") (.var "c"))
  , .declInit f3 "lam" (.index (.var "lambda") (.var "c"))
  , .declInit f  "Cx"  (.bin "-" (.member (.var "p") "x") (.member (.var "fp") "x"))
  , .declInit f  "Cy"  (.bin "-" (.member (.var "p") "y") (.member (.var "fp") "y"))
  , .declInit f  "Cz"  (.bin "-" (.member (.var "p") "z") (.member (.var "fp") "z"))
  , .assign (.index (.var "lambda") (.var "c"))
      (.call "float3"
        [ .bin "+" (.member (.var "lam") "x") (.bin "*" (.var "g") (.var "Cx"))
        , .bin "+" (.member (.var "lam") "y") (.bin "*" (.var "g") (.var "Cy"))
        , .bin "+" (.member (.var "lam") "z") (.bin "*" (.var "g") (.var "Cz")) ])
  -- Eq. 16: ramp the penalty by the constraint violation.
  --
  -- Upstream clamps to `min(stiffness, PENALTY_MAX)` and, for a finite
  -- stiffness, does NOT update lambda -- a soft constraint is meant to
  -- converge to its material stiffness, not to C = 0. This kernel
  -- ascends lambda unconditionally, which makes an attachment an
  -- effectively HARD constraint, so the matching clamp is the hard
  -- ceiling alone. Clamping to the material stiffness here would also
  -- be a no-op: the host seeds gamma from `stiffness`, so gamma would
  -- start already at the cap.
  --
  -- Adopting upstream's soft-constraint rule is a modelling change (it
  -- would let pinned vertices sag by mg/k) and is deliberately not made
  -- here; it should be measured on its own.
  --
  -- Upstream also keeps a per-axis penalty vector and uses abs(C)
  -- componentwise. gamma here is one scalar per constraint, so the
  -- matching reduction is the magnitude |C|.
  , .declInit f  "Cmag"
      (.call "sqrt"
        [ .bin "+"
            (.bin "+" (.bin "*" (.var "Cx") (.var "Cx"))
                      (.bin "*" (.var "Cy") (.var "Cy")))
            (.bin "*" (.var "Cz") (.var "Cz")) ])
  , .assign (.index (.var "gamma") (.var "c"))
      (.call "min"
        [ .bin "+" (.var "g")
            (.bin "*" (.member (.var "params") "beta") (.var "Cmag"))
        , .member (.var "params") "penaltyMax" ])
  ]

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "AttachmentDualUpdateParams"
        , fields := [⟨"beta", f, Semantic.none, none, none, .qIn⟩
            , ⟨"penaltyMax", f, Semantic.none, none, none, .qIn⟩
            , ⟨"count", u, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ bnd 0 "positions" (.roBuf f3)
      , bnd 1 "vertIdx"   (.roBuf u)
      , bnd 2 "fixedPos"  (.roBuf f3)
      , bnd 3 "gamma"     (.rwBuf f)
      , bnd 4 "lambda"    (.rwBuf f3)
      , ⟨"params", .const "AttachmentDualUpdateParams", Semantic.none, some 5, some 0, .qIn⟩
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
"struct AttachmentDualUpdateParams {
  float beta;
  float penaltyMax;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> vertIdx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> fixedPos;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> gamma;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> lambda;
[[vk::binding(5, 0)]]
ConstantBuffer<AttachmentDualUpdateParams> params;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  if ((c >= params.count)) {
    return;
  }
  uint v = vertIdx[c];
  float3 p = positions[v];
  float3 fp = fixedPos[c];
  float g = gamma[c];
  float3 lam = lambda[c];
  float Cx = (p.x - fp.x);
  float Cy = (p.y - fp.y);
  float Cz = (p.z - fp.z);
  lambda[c] = float3((lam.x + (g * Cx)), (lam.y + (g * Cy)), (lam.z + (g * Cz)));
  float Cmag = sqrt((((Cx * Cx) + (Cy * Cy)) + (Cz * Cz)));
  gamma[c] = min((g + (params.beta * Cmag)), params.penaltyMax);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.AttachmentDualUpdate
