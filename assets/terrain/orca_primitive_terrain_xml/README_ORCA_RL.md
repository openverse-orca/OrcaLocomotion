# Orca Primitive Terrain XML

A standalone terrain-only MuJoCo XML package for OrcaLab XML import.

This version intentionally uses only primitive geoms (`plane`, `box`, `cylinder`) and one material.
It removes robot includes, mesh references, hfield PNG assets, and nested helper XML files so the Orca XML importer has a minimal path.

Entry point: `terrain.xml`
