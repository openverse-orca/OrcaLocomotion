# Mjlab Rough 5x5 Terrain XML

A standalone terrain-only MuJoCo XML package for OrcaLab XML import and
`orca_rl.run_play --local-terrain-map` testing.

Layout:

- 5 rows x 5 columns, matching the fixed mjlab rough play layout.
- Rows increase in difficulty along +X.
- Columns mix low blocks, rough blocks, wave bars, ramps, gaps, stairs, and stepping stones.
- The first row is intentionally mild so a single Go2 can spawn and walk into harder tiles.

Files:

- `terrain.xml`: primitive-only MuJoCo terrain. It uses only `plane`, `box`, and `light`.
- `terrain_height_field.npz`: approximate height scan alignment for local MuJoCo play.
- `generate_terrain.py`: deterministic generator used to rebuild both files.

Upload package:

`assets/terrain/MjlabRough5x5Xml.zip`
