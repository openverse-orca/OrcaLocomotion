"""RSL-RL task and vector-environment adapters for Orca locomotion."""

__all__ = ["OrcaRslRlVecEnv", "make_locomotion_vec_env"]


def __getattr__(name: str):
    if name == "OrcaRslRlVecEnv":
        from .adapters import OrcaRslRlVecEnv

        return OrcaRslRlVecEnv
    if name == "make_locomotion_vec_env":
        from .adapters import make_locomotion_vec_env

        return make_locomotion_vec_env
    raise AttributeError(name)
