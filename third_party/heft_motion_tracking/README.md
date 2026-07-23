# HEFT G1 runtime assets

The G1 PMG ONNX policy and the three reference motions in this repository are
copied from the `sim2real` branch of
<https://github.com/Axellwppr/motion_tracking> at commit
`0d5ba31e33397f3543d350d98b637e26d92f470a`.

The upstream project is MIT licensed. See [LICENSE](LICENSE).

Vendored files:

- `checkpoints/heft/G1_PMG/policy.json`
- `checkpoints/heft/G1_PMG/policy.onnx`
- `checkpoints/heft/G1_PMG/policy.onnx.data`
- `assets/heft/motions/walk1_subject1.npz`
- `assets/heft/motions/walk2_subject1.npz`
- `assets/heft/motions/walk3_subject1.npz`

The OrcaLab adapter is an in-process implementation. It does not launch the
upstream UDP `deploy.py` / `sim2sim.py` pair and does not start
`/home/user/unitree-orca`.
