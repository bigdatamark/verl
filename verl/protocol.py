# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [MODULE DOC]
# Core data-transfer protocol of verl. DataProto bundles a TensorDict of batch-major tensors (batch), a mapping of
# batch-major numpy object arrays (non_tensor_batch) and a free-form meta_info mapping, with indexing, selection,
# chunk/concat, repeat, padding, (de)serialization and iteration utilities. Also provides DataProtoFuture (lazy result of
# remote calls), BatchData (type-uniform chunk/concat wrapper used by dispatch), auto-padding configuration, helpers for
# padding/unpadding, unions, batch-dim folding, a numpy-based tensor serialization format and a distributed all-gather.
# At import, tensordict is configured (lazy legacy off; list-to-stack on for tensordict below 0.10), errors suppressed.
# Module constant _padding_size_key is the reserved kwarg name dispatch uses to pass padding counts.

"""
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import contextlib
import copy
import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import ray
import tensordict
import torch
import torch.distributed
from packaging import version
from packaging.version import parse as parse_version
from tensordict import TensorDict
from torch.utils.data import DataLoader

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.py_functional import list_of_dict_to_dict_of_list, union_two_dict
from verl.utils.torch_functional import allgather_dict_tensors
from verl.utils.transferqueue_utils import BatchMeta, KVBatchMeta

__all__ = ["DataProto", "union_tensor_dict"]

with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()
    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        tensordict.set_list_to_stack(True).set()


class _DataProtoConfigMeta(type):
    # [CLASS DOC]
    # HELPER: Metaclass holding a class-level config mapping, giving DataProtoConfig a class property auto_padding.
    _config = {}

    auto_padding_key = "_verl_auto_padding"

    @property
    def auto_padding(cls):
        # [FUNCTION DOC]
        # Outputs: True if env VERL_AUTO_PADDING is TRUE or 1 (case-insensitive) or the config flag is set.
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        # [FUNCTION DOC]
        # Inputs: enabled - bool. Exceptions: AssertionError if not bool. Side effects: sets the global config flag.
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    # [CLASS DOC]
    # Global configuration holder; read/write DataProtoConfig.auto_padding to toggle auto padding for all DataProto.
    pass


_padding_size_key = "_padding_size_key_x123d"


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    # [FUNCTION DOC]
    # Behavior: if length is not a multiple of size_divisor, append copies of leading slices of the data (taking up to the
    # full length repeatedly) until the padding amount is reached, via concat; else return data unchanged (warning if empty).
    # Inputs: data - DataProto; size_divisor - positive int.
    # Outputs: (padded DataProto, pad_size).
    # Exceptions: AssertionError if not a DataProto.
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no changed made")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    # [FUNCTION DOC]
    # Outputs: data with the last pad_size items removed (unchanged if pad_size is 0).
    """Unpad the data proto with pad_size. i.e. `data[:-pad_size]`"""
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    # [FUNCTION DOC]
    # Behavior: add keys of the second TensorDict into the first; shared keys must hold equal tensors.
    # Outputs: the first TensorDict (mutated). Exceptions: AssertionError on batch size mismatch or unequal shared keys.
    """Union two tensordicts."""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(tensor_dict2[key]), (
                f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
            )

    return tensor_dict1


def _array_equal(array1: np.ndarray, array2: np.ndarray, visited: set[int]) -> bool:
    # [FUNCTION DOC]
    # HELPER: Behavior: strict array equality: dtype and shape must match; non-object dtypes use numpy equality treating NaN
    # as equal; object arrays compare element-wise via _deep_equal.
    # Inputs: two arrays; visited - set of object ids on the current recursion path. Outputs: bool.
    """
    Recursively compares two NumPy arrays for strict equality, with special
    handling for object-dtype arrays, NaN values, and circular references.
    This function assumes that the two arguments provided are NumPy arrays.

    Args:
        array1: The first NumPy array.
        array2: The second NumPy array.

    Returns:
        True if the arrays' dtypes, shapes, and all elements are equal.
    """
    # Check dtype and shape first, as this is the fastest failure path.
    if array1.dtype != array2.dtype or array1.shape != array2.shape:
        return False

    # For non-object dtypes, use NumPy's implementation with equal_nan=True.
    if array1.dtype != "object":
        return np.array_equal(array1, array2, equal_nan=True)

    # For object-dtype arrays, we must recursively compare each element.
    # We delegate to _deep_equal to handle elements, as they could be any
    # type, including other nested arrays or NaNs.
    return all(_deep_equal(x, y, visited) for x, y in zip(array1.flat, array2.flat, strict=False))


def _deep_equal(a: Any, b: Any, visited: set[int]) -> bool:
    # [FUNCTION DOC]
    # HELPER: Behavior: deep equality: different exact types are unequal; an object id already on the recursion path counts
    # as equal (cycle); NaN floats equal each other; ndarrays use _array_equal; otherwise plain equality. Maintains visited
    # on entry/exit.
    # Outputs: result of the comparison (bool, or whatever == returns for exotic types).
    """
    Recursively performs a deep comparison between two Python objects.
    - Handles NaN values correctly (NaN == NaN evaluates to True).
    - Handling circular references.
    - Dispatches to _array_equal if both objects are NumPy arrays.
    - Otherwise, uses standard '==' comparison.
    """
    if type(a) is not type(b):
        return False

    # If we have seen this object ID before on this path, it's a cycle.
    # Since we already know the types match, we can safely assume this part
    # of the structure is equal.
    obj_id = id(a)
    if obj_id in visited:
        return True

    visited.add(obj_id)

    # Perform the specific comparison based on type
    result = False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        result = True
    elif isinstance(a, np.ndarray):
        # We know b is also an ndarray due to the initial type check
        result = _array_equal(a, b, visited)
    else:
        # Standard equality for all other types
        result = a == b

    # Clean up the visited set on the way out of the recursion
    visited.remove(obj_id)
    return result


def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # [FUNCTION DOC]
    # Behavior: merge the second mapping of numpy arrays into the first; shared keys must both be arrays and deep-equal.
    # Outputs: the first mapping (mutated). Exceptions: AssertionError on conflicts.
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (
                f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."
            )
        tensor_dict1[key] = val

    return tensor_dict1


def fold_batch_dim(data: "DataProto", new_batch_size):
    # [FUNCTION DOC]
    # Behavior: reshape the batch from (bsz, ...) to (new_bsz, bsz/new_bsz, ...): tensor batch viewed and given 1 batch dim,
    # non-tensor arrays reshaped likewise.
    # Inputs: data; new_batch_size - must divide current batch size. Outputs: new object of the same type, sharing meta_info.
    # Exceptions: AssertionError if not divisible. Side effects: overwrites entries of the input's non_tensor_batch mapping.
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx]
    """
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, (new_batch_size, -1, *val.shape[1:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: "DataProto", batch_dims=2):
    # [FUNCTION DOC]
    # Behavior: treat the first batch_dims dims as batch and flatten them into one; non-tensor arrays reshaped to match.
    # Inputs: data; batch_dims (default 2). Outputs: new object of the same type.
    # Side effects: sets auto batch size on the input's TensorDict in place.
    """
    Unfold the first n dims as new batch dim
    """
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)

    batch_size = tensor.batch_size[0]

    non_tensor_new = {}

    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, (batch_size, *val.shape[batch_dims:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


def serialize_single_tensor(obj: torch.Tensor) -> tuple[str, tuple[int, ...], int | memoryview]:
    # [FUNCTION DOC]
    # HELPER: Outputs: (dtype name without torch prefix, shape, flattened contiguous bytes as a uint8 numpy array). Tensor
    # must be on CPU.
    data = obj.flatten().contiguous().view(torch.uint8).numpy()
    dtype = str(obj.dtype).removeprefix("torch.")
    return dtype, obj.shape, data


def serialize_tensordict(batch: TensorDict) -> tuple[tuple[int, ...], Optional[str], dict[str, tuple[str, Any]]]:
    # [FUNCTION DOC]
    # HELPER: Behavior: encode each entry: regular tensors via serialize_single_tensor (3-tuple); nested tensors as
    # (layout name, list of encoded components).
    # Outputs: (batch size tuple, device string or None, mapping key -> encoding).
    encoded_items: dict[str, tuple[Any]] = {}
    for k, v in batch.items():
        if not v.is_nested:
            encoded_items[k] = serialize_single_tensor(v)
        else:
            layout = str(v.layout).removeprefix("torch.")
            data = [serialize_single_tensor(tensor) for tensor in v.unbind()]
            encoded_items[k] = (layout, data)

    batch_size = tuple(batch.batch_size)
    device = str(batch.device) if batch.device is not None else None
    return batch_size, device, encoded_items


def deserialize_single_tensor(arr: Any) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Behavior: copy bytes to a writable buffer, view as uint8 then reinterpret as the named dtype and reshape.
    # Outputs: tensor. Exceptions: AssertionError if the dtype name is not a torch dtype.
    dtype, shape, data = arr

    torch_dtype = getattr(torch, dtype)
    assert isinstance(torch_dtype, torch.dtype)

    buffer = bytearray(data)
    # Create uint8 array
    arr = torch.frombuffer(buffer, dtype=torch.uint8)
    # Convert back to proper shape & type
    return arr.view(torch_dtype).view(shape)


def deserialize_tensordict(arr: Any) -> TensorDict:
    # [FUNCTION DOC]
    # HELPER: Behavior: inverse of serialize_tensordict; 3-tuples are plain tensors, 2-tuples nested tensors rebuilt with
    # the named layout. Outputs: TensorDict with the batch size and device.
    # Exceptions: ValueError for other encoding lengths.
    batch_size, device, encoded_items = arr
    decoded_items: dict[str, Any] = {}

    for k, v in encoded_items.items():
        if len(v) == 3:
            # decode single tensor
            decoded_items[k] = deserialize_single_tensor(v)
        elif len(v) == 2:
            # decode nested tensor
            layout, data = v
            torch_layout = getattr(torch, layout)
            decoded_items[k] = torch.nested.as_nested_tensor(
                [deserialize_single_tensor(tensor) for tensor in data], layout=torch_layout
            )
        else:
            raise ValueError(f"Invalid tensor encoding format, expected length 2 or 3, got {len(v)}")

    return TensorDict(source=decoded_items, batch_size=batch_size, device=device)


def collate_fn(x: list["DataProtoItem"]):
    # [FUNCTION DOC]
    # HELPER: DataLoader collate: stack item TensorDicts, convert list of non-tensor dicts into dict of object arrays.
    # Inputs: list of DataProtoItem. Outputs: DataProto (empty meta_info).
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    # [CLASS DOC]
    # Dataclass for a single sample: batch (TensorDict row or None), non_tensor_batch (per-key element), meta_info.
    # TODO(zhangchi.usc1992) add consistency check
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)


@dataclass
class DataProto:
    # [CLASS DOC]
    # Dataclass; see module doc. Invariants (checked at construction): batch has exactly one batch dim; every
    # non_tensor_batch value is a numpy array whose first dim equals the batch size.
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: runs check_consistency. Exceptions: AssertionError on violations.
        # perform necessary checking
        self.check_consistency()

    def __len__(self):
        # [FUNCTION DOC]
        # Outputs: batch size from the tensor batch, else first dim of the first non-tensor array, else 0.
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Behavior: slice -> slice(); list/ndarray/tensor -> select_idxs(); int/numpy integer -> DataProtoItem with row of
        # batch, element of each non-tensor array and shared meta_info.
        # Exceptions: TypeError for other index types.
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            DataProto: For all indexing types except single integers
            DataProtoItem: Only for single integer indices
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, list | np.ndarray | torch.Tensor):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        elif isinstance(item, int | np.integer):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        # [FUNCTION DOC]
        # Behavior: pickling. If env VERL_DATAPROTO_SERIALIZATION_METHOD is numpy, return (serialized tensordict or None,
        # non_tensor_batch, meta_info). Otherwise, for tensordict >= 0.5 with non-empty batch, make contiguous and consolidate;
        # torch.save the batch into bytes and return (bytes, non_tensor_batch, meta_info).
        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            return (
                serialize_tensordict(self.batch) if self.batch is not None else None,
                self.non_tensor_batch,
                self.meta_info,
            )
        else:
            if version.parse(tensordict.__version__) >= version.parse("0.5.0") and self.batch is not None:
                # Check if batch is empty to avoid torch.cat error in consolidate
                if len(self.batch.keys()) > 0:
                    batch = self.batch.contiguous().consolidate()
                else:
                    batch = self.batch
            else:
                batch = self.batch

            import io

            buffer = io.BytesIO()
            torch.save(batch, buffer)
            buffer_bytes = buffer.getvalue()
            return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        # [FUNCTION DOC]
        # Behavior: unpickling counterpart; the numpy path decodes with deserialize_tensordict; the default path torch.loads
        # (weights_only False) mapping to CPU when no accelerator is available. Sets all three fields.
        # Note: the env var must match between pickling and unpickling processes.
        batch_deserialized_bytes, non_tensor_batch, meta_info = data

        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            if batch_deserialized_bytes is not None:
                self.batch = deserialize_tensordict(batch_deserialized_bytes)
            else:
                self.batch = None
        else:
            import io

            batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
            batch = torch.load(
                batch_deserialized,
                weights_only=False,
                map_location="cpu" if not get_torch_device().is_available() else None,
            )
            self.batch = batch

        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath):
        # [FUNCTION DOC]
        # Side effects: pickles self to filepath.
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> "DataProto":
        # [FUNCTION DOC]
        # Static. Outputs: object unpickled from filepath (pickle; trust the file).
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        # [FUNCTION DOC]
        # Behavior: prints total bytes (in GB) of tensor batch and non-tensor arrays, optionally prefixed.
        # Side effects: stdout.
        size_of_tensordict = 0
        if self.batch is not None:
            for _, tensor in self.batch.items():
                size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for _, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        if prefix:
            message = f"{prefix}, " + message
        print(message)

    def check_consistency(self):
        # [FUNCTION DOC]
        # Behavior: asserts the invariants described in the class doc. Exceptions: AssertionError.
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), (
                    f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for "
                    f"{key=}, got {type(val)=}"
                )
                assert val.shape[0] == batch_size, (
                    f"key {key} length {len(val)} is not equal to batch size {batch_size}"
                )

    @classmethod
    def from_single_dict(cls, data: dict[str, torch.Tensor | np.ndarray], meta_info=None, auto_padding=False):
        # [FUNCTION DOC]
        # Behavior: split a mapping into tensors and numpy arrays and build via from_dict.
        # Exceptions: ValueError for values of other types.
        """Create a DataProto from a dict of tensors and non_tensors"""
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors=None,
        meta_info=None,
        num_batch_dims=1,
        auto_padding=False,
    ):
        # [FUNCTION DOC]
        # Behavior: validate num_batch_dims (positive; 1 if non-tensors given), default empties, check all tensors share the
        # leading num_batch_dims dims, wrap non-array non-tensor values into object arrays, build a TensorDict (None if no
        # tensors), and if auto_padding set the auto-padding key in meta_info.
        # Outputs: new instance. Exceptions: AssertionError on violated constraints.
        # Side effects: mutates the passed non_tensors and meta_info mappings.
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """

        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if tensors is None:
            tensors = {}
        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, (
                    f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. "
                    f"Got {pivot_key} has {batch_size}, {key} has {current_batch}"
                )

        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    @classmethod
    def from_tensordict(
        cls,
        tensor_dict: TensorDict = None,
        meta_info=None,
        num_batch_dims=1,
    ):
        # [FUNCTION DOC]
        # Behavior: requires tensordict >= 0.10. Splits a TensorDict: tensors to batch (batch size from the first tensor),
        # NonTensorStack entries to object arrays of their elements, NonTensorData entries to meta_info.
        # Outputs: new instance. Exceptions: AssertionError on version or num_batch_dims constraints.
        """Create a DataProto from a TensorDict. This assumes that
        1. All the tensor in tensor_dict have the same dim0
        2. Only dim0 is the batch dim
        """
        assert version.parse(tensordict.__version__) >= version.parse("0.10.0"), (
            "Build DataProto from TensorDict at least requires tensordict version 0.10.0"
        )
        from tensordict import NonTensorData, NonTensorStack

        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if not all(isinstance(val, torch.Tensor) for val in tensor_dict.values()):
            assert num_batch_dims == 1, "only support num_batch_dims=1 when tensor_dict contains non tensor data."

        if meta_info is None:
            meta_info = {}
        batch = {}
        non_tensor_batch = {}
        batch_size = None
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val
                if batch_size is None:
                    batch_size = val.shape[:num_batch_dims]
            elif isinstance(val, NonTensorStack):
                non_tensor_batch[key] = np.array([elem.data for elem in val], dtype=object)
            elif isinstance(val, NonTensorData):
                meta_info[key] = val.data

        return cls(
            batch=TensorDict(batch, batch_size=batch_size),
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def to(self, device) -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: move the tensor batch to device in place. Outputs: self.
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: build a new instance with only the given batch keys, non-tensor keys and meta keys (None = keep all);
        # optionally deep-copy non-tensor and meta parts. Tensor batch is not copied.
        # Outputs: new instance.
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        # [FUNCTION DOC]
        # Behavior: lists become tensors (int32 unless boolean); numpy and torch indices are converted to both forms. Result
        # size is the number of true entries for boolean masks, else the index count. Tensors are indexed into a new TensorDict
        # on the same device; non-tensor arrays indexed with the numpy indices.
        # Outputs: new instance sharing meta_info.
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = int(idxs_np.sum()) if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()},
                batch_size=(batch_size,),
                device=self.batch.device,
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        # [FUNCTION DOC]
        # Behavior: apply the same slice to the TensorDict and every non-tensor array. Outputs: new instance sharing meta_info.
        """
        Slice the DataProto and return a new DataProto object.
        This is an improved version of direct slicing which returns a DataProtoItem.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index still returns DataProtoItem
            single_item = data_proto[5]
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: remove the named keys from batch, non-tensor batch and meta_info and return them as a new DataProto.
        # Exceptions: AssertionError if a key is missing. Side effects: mutates self.
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        if batch_keys is None:
            batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: rename batch keys in place (strings are treated as single-element lists).
        # Outputs: self. Exceptions: TypeError for invalid key types; ValueError on length mismatch (a None argument fails
        # with TypeError when taking its length).
        """
        Note that this function only rename the key in the batch
        """

        def validate_input(keys):
            # [FUNCTION DOC]
            # HELPER: normalizes a key argument to a list (None passes through).
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: in-place union of tensor batch, non-tensor batch and meta_info with another instance.
        # Outputs: self. Exceptions: AssertionError on conflicts.
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if

        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        # [FUNCTION DOC]
        # Behavior: build a DataLoader over self with the given mini-batch size and collate_fn, optional seeded generator and
        # extra kwargs; return an iterator that loops for the given epochs and attaches self.meta_info to each mini-batch.
        # Exceptions: AssertionError if batch size is not divisible by mini_batch_size.
        # Note: the seed only affects order if shuffle is passed in dataloader_kwargs.
        r"""Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/stable/tutorials/data_fashion for more details.


        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``.
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs (Any): internally, it returns a DataLoader over the batch. The
                dataloader_kwargs is the kwargs passed to the DataLoader.

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration
                steps is ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            # [FUNCTION DOC]
            # HELPER: generator yielding mini-batches for each epoch.
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def is_padding_enabled(self):
        # [FUNCTION DOC]
        # Outputs: True if this instance's meta_info enables auto padding or the global config does.
        """
        Check if padding is enabled for the DataProto.
        Returns:
            bool: True if padding is enabled, False otherwise.
        """
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    def padding(self, padding_size, padding_candidate=""):
        # [FUNCTION DOC]
        # Behavior: no-op for 0; else select the first item (if candidate is first) or the last item, repeat it padding_size
        # times, and concatenate in place.
        # Side effects: replaces self.batch and self.non_tensor_batch.
        # KNOWN ISSUE: the docstring says only first/last are supported, but any other value (including the default empty
        # string) silently pads with the last item.
        """Pad the DataProto by concating with padding_candidate.repeat(padding_size)

        Args:
            padding_size (int): the number of repeated padding_candidate
            padding_candidate: the item to be repeated and appended to the DataProto, only supporting ["first", "last"]
        """
        if padding_size == 0:
            return
        padding_candidate = self.select_idxs([0 if padding_candidate == "first" else len(self) - 1])
        padding_part = padding_candidate.repeat(padding_size)
        padded_dp = DataProto.concat([self, padding_part])
        self.batch = padded_dp.batch
        self.non_tensor_batch = padded_dp.non_tensor_batch

    def chunk(self, chunks: int) -> list["DataProto"]:
        # [FUNCTION DOC]
        # Behavior: unless padding is enabled, assert length divisible by chunks. Chunk the TensorDict along dim 0 and split
        # non-tensor arrays at the same boundaries (or evenly if no tensor batch). Each chunk shares meta_info.
        # Outputs: list of instances. Exceptions: AssertionError.
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, (
                f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."
            )

        bsz_in_batch = None
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            if bsz_in_batch is not None:
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            else:
                non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info)
            )

        return output

    def split(self, split_size: int) -> list["DataProto"]:
        # [FUNCTION DOC]
        # Outputs: list of consecutive slices of length split_size (last may be shorter).
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            split_size (int): the size of each split

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        return [self[i : i + split_size] for i in range(0, len(self), split_size)]

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: concatenate tensor batches (if the first is not None) and non-tensor arrays along dim 0. Merge meta_info:
        # metrics entries from all inputs are collected (lists extended, others appended) and converted from list of dicts to
        # dict of lists; other keys must agree across inputs. Result type is that of the first element.
        # Exceptions: AssertionError on conflicting meta values; IndexError for an empty list.
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is merged, with special handling for metrics from different workers.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        # Merge meta_info with special handling for metrics
        merged_meta_info = {}
        if data:
            # Merge non-metric meta_info and aggregate metrics from all workers.
            all_metrics = []
            for d in data:
                for k, v in d.meta_info.items():
                    if k == "metrics":
                        if v is not None:
                            if isinstance(v, list):
                                all_metrics.extend(v)
                            else:
                                all_metrics.append(v)
                    else:
                        if k in merged_meta_info:
                            # Ensure consistency for overlapping non-metric keys
                            assert merged_meta_info[k] == v, f"Conflicting values for meta_info key '{k}'"
                        else:
                            merged_meta_info[k] = v

            # Flatten list of dicts to dict of lists for consistent metrics structure
            if all_metrics:
                merged_meta_info["metrics"] = list_of_dict_to_dict_of_list(all_metrics)

        cls = type(data[0]) if len(data) > 0 else DataProto
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=merged_meta_info)

    def reorder(self, indices):
        # [FUNCTION DOC]
        # Behavior: in-place reorder of batch and non-tensor arrays by an index tensor.
        # Note: the tensor must be on CPU (converted to numpy).
        """
        Note that this operation is in-place
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True):
        # [FUNCTION DOC]
        # Behavior: repeat every sample repeat_times: interleaved (each sample repeated consecutively) or tiled (whole batch
        # repeated). Applies to tensors and non-tensor arrays. Outputs: new instance sharing meta_info.
        # Note: the new TensorDict is created without a device.
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def unfold_column_chunks(self, n_split: int, split_keys: Optional[list[str]] = None):
        # [FUNCTION DOC]
        # Behavior: for keys in split_keys, reshape (B, S, ...) to (B*n_split, S/n_split, ...); for other keys repeat each row
        # n_split times (interleaved). Same for non-tensor arrays.
        # Outputs: new instance sharing meta_info.
        # KNOWN ISSUE: the docstring says split_keys may be omitted, but the non-tensor loop tests membership in split_keys
        # directly and fails with TypeError when it is None and non-tensor data exists; also the tensor branch treats None as
        # repeat-all, contrary to the docstring wording.
        """Split along the second dim into `n_split`, unfold it to the first dim (batch dim)
        Useful in passing grouped tensors that doesn't want to be shuffled in dataset.
        keys not in split_keys are repeated to match the shape
        Note that if the `split_keys` is not provided, it will repeat all the keys in the second dim.
        """
        if self.batch is not None:
            unfolded_batch = {}
            for key in self.batch.keys():
                if key in split_keys if split_keys is not None else False:
                    shape = list(self.batch[key].shape)
                    shape[0] = self.batch[key].shape[0] * n_split
                    shape[1] = self.batch[key].shape[1] // n_split
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                else:
                    unfolded_batch[key] = torch.repeat_interleave(self.batch[key], n_split, dim=0)
            # locate the `unfolded_batch` as a TensorDict on the same device as the original batch
            unfolded_batch = TensorDict(
                source=unfolded_batch, batch_size=(self.batch.batch_size[0] * n_split,), device=self.batch.device
            )
        else:
            unfolded_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                shape = list(val.shape)
                shape[0] = val.shape[0] * n_split
                shape[1] = val.shape[1] // n_split
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            else:
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)

        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def sample_level_repeat(self, repeat_times):
        # [FUNCTION DOC]
        # Behavior: normalize repeat_times (tuple, 1D tensor, 1D array or list) to a tensor and repeat each sample its own
        # number of times for tensors and non-tensor arrays. Outputs: new instance on the same device.
        # Exceptions: AssertionError for wrong types or shapes.
        """
        Repeat each row of the batch data a specified number of times.

        Args:
            repeat_times (torch.tensor, list, tuple, ndarray):  Number of times to repeat the data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        elif isinstance(repeat_times, torch.Tensor):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        elif isinstance(repeat_times, np.ndarray):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        else:
            assert isinstance(repeat_times, list), (
                f"repeat_times type must be in [list, torch.Tensor, np.ndarray, tuple], got {type(repeat_times)}"
            )
        repeat_times = torch.tensor(repeat_times)

        if self.batch is not None:
            # Interleave the data
            repeated_tensors = {
                key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
            }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def to_tensordict(self) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: requires tensordict >= 0.10; convert batch to a plain dict, add each non-tensor array as a NonTensorStack of
        # NonTensorData items, and build a TensorDict with meta_info as non-tensor entries via tensordict_utils.get_tensordict.
        # Outputs: TensorDict. Exceptions: AssertionError on version or overlapping keys.
        """Convert this DataProto to TensorDict. Note that this requires tensordict version at least 0.10

        Returns:

        """
        assert parse_version(tensordict.__version__) >= parse_version("0.10"), (
            "Convert DataProto to TensorDict at least requires tensordict version 0.10"
        )
        tensor_batch = self.batch.to_dict()
        non_tensor_batch = self.non_tensor_batch

        from tensordict.tensorclass import NonTensorData, NonTensorStack

        from verl.utils import tensordict_utils as tu

        common_keys = set(tensor_batch.keys()) & set(non_tensor_batch.keys())
        assert len(common_keys) == 0, f"tensor_batch and non_tensor_batch have common keys {common_keys}"

        for key, val in non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            # Convert to NonTensorStack instead of plain list to handle nested structures
            tensor_batch[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])
        output = tu.get_tensordict(tensor_dict=tensor_batch, non_tensor_dict=self.meta_info)
        return output

    def get_data_info(self) -> str:
        # [FUNCTION DOC]
        # Outputs: multi-line string describing batch entries (shape, dtype, device), non-tensor arrays and recursive meta types.
        # Exceptions: fails if batch is None.
        """Return formatted information about stored data with nested type details.

        Returns:
            str: Formatted string showing tensor details and recursive metadata types
        """
        info = ["batch"]

        for key, tensor in self.batch.items():
            if hasattr(tensor, "shape") and hasattr(tensor, "dtype") and hasattr(tensor, "device"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype}) {tensor.device}")
            elif hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
            else:
                info.append(f"  {key}: {type(tensor).__name__}")

        info.append("non_tensor_batch")
        for key, array in self.non_tensor_batch.items():
            info.append(f"  {key}: ndarray{array.shape} ({array.dtype})")

        info.append("meta_info")
        for k, v in self.meta_info.items():
            type_info = self._get_type_info(v)
            info.append(f"  {k}: {type_info}")

        return "\n".join(info)

    def _get_type_info(self, value):
        # [FUNCTION DOC]
        # HELPER: Outputs: string type description: list from the types of up to 3 elements, tuple of all elements, dict from the
        # first item, ndarray with shape/dtype, else the type name.
        """Recursively get type information for nested structures"""
        if isinstance(value, list):
            elem_types = {self._get_type_info(v) for v in value[:3]}
            return f"list[{'|'.join(elem_types) if elem_types else '...'}]"
        if isinstance(value, tuple):
            elem_types = [self._get_type_info(v) for v in value]
            return f"tuple({', '.join(elem_types)})"
        if isinstance(value, dict):
            if not value:
                return "dict"
            k, v = next(iter(value.items()))
            return f"dict[{self._get_type_info(k)}: {self._get_type_info(v)}]"
        if isinstance(value, np.ndarray):
            return f"ndarray{value.shape} ({value.dtype})"
        return type(value).__name__


@dataclass
class DataProtoFuture:
    # [CLASS DOC]
    # Dataclass: collect_fn (reduces fetched outputs), futures (Ray object refs), optional dispatch_fn (selects this
    # consumer's share after collection).
    """
    DataProtoFuture aims to eliminate actual data fetching on driver. By doing so, the driver doesn't have to wait
    for data so that asynchronous execution becomes possible.
    DataProtoFuture contains a list of futures from another WorkerGroup of size world_size.
    - collect_fn is a Callable that reduces the list of futures to a DataProto
    - dispatch_fn is a Callable that partitions the DataProto into a list of DataProto of size world_size
        and then select

    Potential issue: we can optimize dispatch_fn(collect_fn) such that only needed data is fetched on destination
    - DataProtoFuture only supports directly passing from the output of a method to another input. You can't perform any
    operation on the DataProtoFuture in driver.
    """

    collect_fn: Callable
    futures: list[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: list[ray.ObjectRef]) -> "DataProtoFuture":
        # [FUNCTION DOC]
        # Static. Outputs: future that concatenates the given refs with DataProto.concat.
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> list["DataProtoFuture"]:
        # [FUNCTION DOC]
        # Behavior: build chunks futures sharing the same refs and collect_fn, each with a dispatch_fn selecting chunk i of the
        # collected result. Outputs: list of futures.
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                # [FUNCTION DOC]
                # HELPER: returns the i-th of chunks pieces of x.
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(
                collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures
            )
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        # [FUNCTION DOC]
        # Behavior: ray.get all refs; concatenate DataProto results via DataProto.concat or TensorDict results via
        # concat_tensordict (collect_fn is not used); apply dispatch_fn if set.
        # Outputs: DataProto or TensorDict. Exceptions: AssertionError for other types.
        # KNOWN ISSUE: the stored collect_fn is ignored; and the TypeError branch is unreachable (and would reference the wrong
        # variable) because of the preceding assertion.
        output = ray.get(self.futures)  # dp_size.
        for o in output:
            assert isinstance(o, DataProto | TensorDict)

        if isinstance(output[0], DataProto):
            output = DataProto.concat(output)  # select dp, concat
        elif isinstance(output[0], TensorDict):
            from verl.utils.tensordict_utils import concat_tensordict

            output = concat_tensordict(output)
        else:
            raise TypeError(f"Unknown type {type(o[0])} in DataProtoFuture")

        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        return output


class BatchData:
    # [CLASS DOC]
    # Wrapper dispatching chunk/concat by concrete type (DataProto, DataProtoFuture, TensorDict, KVBatchMeta, BatchMeta, Ray
    # refs). State: the wrapped data. Class tuples of types are placeholders; the lazy classmethods return the full sets.
    """Uniform dispatch wrapper for batch data operations.

    All type-specific logic (isinstance checks) is centralized here so that
    callers (e.g. decorator.py) never need to branch on the concrete data type.

    Usage::

        # chunk a single data item into N pieces
        chunks = BatchData(arg).chunk(chunks=N)

        # concat a list of data items into one
        merged = BatchData(output_list).concat()

        # validate before dispatching
        assert BatchData(arg).is_chunkable()
        assert BatchData(output_list).is_concatable()
    """

    _CHUNKABLE_TYPES = (TensorDict, KVBatchMeta, BatchMeta)  # lazily extended with DataProto etc.
    _CONCATABLE_TYPES = (TensorDict, KVBatchMeta, BatchMeta)

    def __init__(self, data):
        # [FUNCTION DOC]
        # Inputs: data - single item (for chunk) or list (for concat).
        self._data = data

    # ---- validation ----------------------------------------------------------

    def is_chunkable(self) -> bool:
        # [FUNCTION DOC]
        # Outputs: True if the data is a DataProto, DataProtoFuture, TensorDict, KVBatchMeta or BatchMeta.
        """Return True if the wrapped data supports chunk dispatch."""
        return isinstance(self._data, self._chunkable_types())

    def is_concatable(self) -> bool:
        # [FUNCTION DOC]
        # Outputs: True if the data is a non-empty list/tuple whose first element is a DataProto, Ray ref, TensorDict,
        # KVBatchMeta or BatchMeta.
        """Return True if the wrapped list of data supports concat collect."""
        data = self._data
        if not isinstance(data, list | tuple) or len(data) == 0:
            return False
        return isinstance(data[0], self._concatable_types())

    # ---- operations ----------------------------------------------------------

    def chunk(self, chunks: int):
        # [FUNCTION DOC]
        # Behavior: TensorDict -> chunk via tensordict utils, make each contiguous and consolidated, returned as a tuple;
        # KVBatchMeta is first converted to BatchMeta; others call their own chunk method.
        # Outputs: sequence of chunks of the original type.
        """Split the wrapped data into *chunks* pieces along the batch dim.

        Returns a tuple/list of the **original data types** (not BatchData).
        """
        data = self._data
        if isinstance(data, TensorDict):
            from verl.utils.tensordict_utils import chunk_tensordict, contiguous

            raw_chunks = chunk_tensordict(data, chunks)
            return tuple(contiguous(val).consolidate() for val in raw_chunks)
        if isinstance(data, KVBatchMeta):
            # early translate KVBatchMeta -> BatchMeta to prevent frequent
            # controller communication during PUT/GET in each rank
            from verl.utils.transferqueue_utils import kv_batch_meta2batch_meta

            data = kv_batch_meta2batch_meta(data)
        # DataProto, DataProtoFuture, etc. all expose .chunk()
        return data.chunk(chunks=chunks)

    def concat(self):
        # [FUNCTION DOC]
        # Behavior: Ray refs -> DataProtoFuture.concat; TensorDict -> concat_tensordict; BatchMeta -> merge extra_info values per
        # key into lists (clearing each input's extra_info), concatenate, attach merged extra_info and convert to KVBatchMeta;
        # otherwise call the element type's concat.
        # Exceptions: ValueError for an empty list. Side effects: clears extra_info of input BatchMeta objects.
        """Concat the wrapped list of data items into a single result.

        Returns the **original data type** (not BatchData).
        """
        data = self._data
        if not data:
            raise ValueError("Cannot concatenate an empty list of data items.")
        sample = data[0]
        if isinstance(sample, ray.ObjectRef):
            return DataProtoFuture.concat(data)
        if isinstance(sample, TensorDict):
            from verl.utils.tensordict_utils import concat_tensordict

            return concat_tensordict(data)
        if isinstance(sample, BatchMeta):
            # handle extra_info merge manually
            merged_extra_info = {}
            for meta in data:
                for k, v in meta.extra_info.items():
                    merged_extra_info.setdefault(k, []).append(v)
                meta.extra_info = {}

            # translate BatchMeta -> KVBatchMeta
            batch_meta = BatchMeta.concat(data)
            batch_meta.extra_info = merged_extra_info
            from verl.utils.transferqueue_utils import batch_meta2kv_batch_meta

            return batch_meta2kv_batch_meta(batch_meta)
        # DataProto, etc. expose .concat() as classmethod / staticmethod
        return type(sample).concat(data)

    # ---- helpers (lazy type tuples to avoid import-order issues) -------------

    @classmethod
    def _chunkable_types(cls):
        # [FUNCTION DOC]
        # HELPER: Outputs: tuple of chunkable types.
        return (DataProto, DataProtoFuture, TensorDict, KVBatchMeta, BatchMeta)

    @classmethod
    def _concatable_types(cls):
        # [FUNCTION DOC]
        # HELPER: Outputs: tuple of concatable element types.
        return (DataProto, ray.ObjectRef, TensorDict, KVBatchMeta, BatchMeta)


def all_gather_data_proto(data: DataProto, process_group):
    # [FUNCTION DOC]
    # Behavior: in place, all-gather a DataProto across process_group: move batch to the current accelerator, all-gather
    # tensors along dim 0, move back to the original device, all-gather non-tensor mappings as objects and concatenate per
    # key (in rank order).
    # Inputs: data - DataProto with a tensor batch; process_group. Outputs: none.
    # Side effects: mutates data; collective communication.
    # Note that this is an inplace operator just like torch.distributed.all_gather
    group_size = torch.distributed.get_world_size(group=process_group)
    assert isinstance(data, DataProto)
    prev_device = data.batch.device
    data = data.to(get_device_id())
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    data = data.to(prev_device)
    # all gather non_tensor_batch
    all_non_tensor_batch = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
