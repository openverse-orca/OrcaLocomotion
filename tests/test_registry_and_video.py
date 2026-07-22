import numpy as np

from orcalab_rslrl.recording import WandbVideoRecorder
from orcalab_rslrl.orca import list_tasks, register_task
from orcalab_rslrl.tasks.registry import make_task


def test_registry():
    register_task("test-task", lambda value=1: value, "test")
    assert make_task("test-task", value=7) == 7
    assert list_tasks()[-1].description == "test"


def test_video_validates_and_buffers_frames():
    rec = WandbVideoRecorder(lambda: np.zeros((24, 32, 3), dtype=np.uint8))
    rec.capture()
    assert len(rec.frames) == 1 and rec.frames[0].shape == (24, 32, 3)
