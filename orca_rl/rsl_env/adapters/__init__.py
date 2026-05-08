from __future__ import annotations

__all__ = [
    "OrcaRslRlVecEnv",
    "make_locomotion_vec_env",
]


def __getattr__(name: str):
    if name == "make_locomotion_vec_env":
        from .factory import make_locomotion_vec_env

        return make_locomotion_vec_env
    if name == "OrcaRslRlVecEnv":
        from .vecenv import OrcaRslRlVecEnv

        return OrcaRslRlVecEnv
    raise AttributeError(name)
