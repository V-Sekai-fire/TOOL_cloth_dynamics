import Cloth.Avbd.CsrProperties
import Cloth.Avbd.ColoringProperties

/-- Falsifiability runner for the AVBD Lean specs.

Each check searches for a mesh on which a deliberately broken
implementation is CAUGHT by the corresponding property. A
`provablyNone` outcome here is a failure: it would mean the property
is blind and the invariant it claims to guard is unguarded. -/
def main : IO UInt32 := do
  Cloth.Avbd.CsrProperties.checkFalsifiability
  Cloth.Avbd.ColoringProperties.checkFalsifiability
  pure 0
