"""Accelerator selection and the launcher's GPU masks, for CUDA and Intel XPU builds of torch.

These run without a GPU: the torch build is faked, so CPU CI covers the XPU branches too.
"""

import importlib

import pytest
import torch

import prime_rl.utils.device as device
from prime_rl.utils.process import (
    DEFAULT_TRAINER_ENV_VARS,
    get_physical_gpu_ids,
    local_trainer_env_vars,
    partition_gpus,
)


@pytest.fixture
def select_build(monkeypatch):
    """Re-select the accelerator as if torch were built for ``accelerator`` (``None``: CPU-only)."""

    def _select(accelerator: str | None):
        current = None if accelerator is None else torch.device(accelerator)
        monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: current)
        return importlib.reload(device)

    yield _select
    monkeypatch.undo()
    importlib.reload(device)


@pytest.mark.parametrize(
    "accelerator, device_type, backend, visible_devices_env, module",
    [
        ("xpu", "xpu", "xccl", "ZE_AFFINITY_MASK", torch.xpu),
        ("cuda", "cuda", "nccl", "CUDA_VISIBLE_DEVICES", torch.cuda),
        # CPU-only and other builds keep the CUDA defaults, so dry runs and CPU tests are unchanged.
        ("mps", "cuda", "nccl", "CUDA_VISIBLE_DEVICES", torch.cuda),
        (None, "cuda", "nccl", "CUDA_VISIBLE_DEVICES", torch.cuda),
    ],
)
def test_accelerator_selection(select_build, accelerator, device_type, backend, visible_devices_env, module):
    selected = select_build(accelerator)
    assert selected.DEVICE_TYPE == device_type
    assert selected.DIST_BACKEND == backend
    assert selected.VISIBLE_DEVICES_ENV == visible_devices_env
    assert selected.device_module is module


def test_xpu_mask_keeps_tile_selectors(select_build, monkeypatch):
    select_build("xpu")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0.0, 0.1,1")
    # CUDA_VISIBLE_DEVICES is not read on XPU.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert get_physical_gpu_ids() == ["0.0", "0.1", "1"]
    assert partition_gpus(num_infer=1, num_train=2) == (["0.0"], ["0.1", "1"])


@pytest.mark.parametrize("mask", [None, "", "  ", "default", "Default"])
def test_xpu_without_a_mask_sees_every_device(select_build, monkeypatch, mask):
    select_build("xpu")
    if mask is None:
        monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    else:
        monkeypatch.setenv("ZE_AFFINITY_MASK", mask)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 3)
    assert get_physical_gpu_ids() == [0, 1, 2]


def test_cuda_mask_is_read_as_indices(select_build, monkeypatch):
    select_build("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3")
    # ZE_AFFINITY_MASK is not read on CUDA.
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0.0")
    assert get_physical_gpu_ids() == [2, 3]
    assert partition_gpus(num_infer=1, num_train=1) == ([2], [3])


def test_partition_rejects_more_gpus_than_the_mask_exposes(select_build, monkeypatch):
    select_build("xpu")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1")
    with pytest.raises(ValueError, match="only 2 physical GPU"):
        partition_gpus(num_infer=1, num_train=2)


def test_xpu_trainer_turns_expandable_segments_off(select_build):
    select_build("xpu")
    env = local_trainer_env_vars()
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
    assert {key: value for key, value in env.items() if key != "PYTORCH_CUDA_ALLOC_CONF"} == {
        key: value for key, value in DEFAULT_TRAINER_ENV_VARS.items() if key != "PYTORCH_CUDA_ALLOC_CONF"
    }


def test_cuda_trainer_keeps_the_default_env(select_build):
    select_build("cuda")
    assert local_trainer_env_vars() == DEFAULT_TRAINER_ENV_VARS
