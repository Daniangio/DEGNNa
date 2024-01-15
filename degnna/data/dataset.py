import logging
import inspect
import yaml
import hashlib
import degnna
import torch

import numpy as np
import MDAnalysis as mda

from os.path import dirname, abspath, basename
from typing import Tuple, Dict, Any, List, Union, Optional

from degnna.utils.torch_geometric import Batch, Dataset
from degnna.utils.torch_geometric.utils import download_url, extract_zip

from degnna.data import (
    AtomicData,
    AtomicDataDict,
)
from degnna.utils.savenload import atomic_write

from .AtomicData import _process_dict
from ase.cell import Cell


class AtomicDataset(Dataset):
    """The base class for all degnna datasets."""

    fixed_fields: Dict[str, Any]
    root: str

    def __init__(
        self,
        root: str,
    ):
        super().__init__(root=root)

    def _get_parameters(self) -> Dict[str, Any]:
        """Get a dict of the parameters used to build this dataset."""
        pnames = list(inspect.signature(self.__init__).parameters)
        IGNORE_KEYS = {
            # the type mapper is applied after saving, not before, so doesn't matter to cache validity
            "type_mapper",
            "dataset_idx"
        }
        params = {
            k: getattr(self, k)
            for k in pnames
            if k not in IGNORE_KEYS and hasattr(self, k)
        }
        # Add other relevant metadata:
        params["dtype"] = str(torch.get_default_dtype())
        params["degnna_version"] = degnna.__version__
        return params

    @property
    def processed_dir(self) -> str:
        # We want the file name to change when the parameters change
        # So, first we get all parameters:
        params = self._get_parameters()
        # Make some kind of string of them:
        # we don't care about this possibly changing between python versions,
        # since a change in python version almost certainly means a change in
        # versions of other things too, and is a good reason to recompute
        buffer = yaml.dump(params).encode("ascii")
        # And hash it:
        param_hash = hashlib.sha1(buffer).hexdigest()
        return f"{self.root}/processed_datasets/processed_dataset_{param_hash}"


class AtomicInMemoryDataset(AtomicDataset):
    r"""Base class for all datasets that fit in memory.

    Please note that, as a ``pytorch_geometric`` dataset, it must be backed by some kind of disk storage.
    By default, the raw file will be stored at root/raw and the processed torch
    file will be at root/process.

    Subclasses must implement:
     - ``get_data()``

    Subclasses may implement:
     - ``download()`` or ``self.url`` or ``ClassName.URL``

    Args:
        root (str, optional): Root directory where the dataset should be saved. Defaults to current working directory.
        url (str, optional): url to download data source
        force_fixed_keys (list, optional): keys to move from AtomicData to fixed_fields dictionary
        extra_fixed_fields (dict, optional): extra key that are not stored in data but needed for AtomicData initialization
        include_frames (list, optional): the frames to process with the constructor.
    """

    def __init__(
        self,
        root: str,
        dataset_idx: int,
        url: Optional[str] = None,
        force_fixed_keys: List[str] = [],
        extra_fixed_fields: Dict[str, Any] = {},
        force_index_keys: List[str] = [AtomicDataDict.EDGE_INDEX_KEY],
        include_frames: Optional[List[int]] = None,
    ):
        self.dataset_idx = dataset_idx
        # TO DO, this may be simplified
        # See if a subclass defines some inputs
        force_fixed_keys = set(force_fixed_keys).union(
            getattr(type(self), "FORCE_FIXED_KEYS", [])
        )
        force_index_keys = set(force_index_keys).union(
            getattr(type(self), "FORCE_INDEX_KEYS", [])
        )
        self.url = getattr(type(self), "URL", url)

        self.force_fixed_keys = force_fixed_keys
        self.extra_fixed_fields = extra_fixed_fields
        self.force_index_keys = force_index_keys
        self.include_frames = include_frames

        self.data = None
        self.fixed_fields = None

        # !!! don't delete this block.
        # otherwise the inherent children class
        # will ignore the download function here
        class_type = type(self)
        if class_type != AtomicInMemoryDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = AtomicInMemoryDataset.download
            if "process" not in self.__class__.__dict__:
                class_type.process = AtomicInMemoryDataset.process

        # Initialize the InMemoryDataset, which runs download and process
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
        # Then pre-process the data if disk files are not found
        super().__init__(root=root)
        if self.data is None:
            self.data, self.fixed_fields, include_frames = torch.load(
                self.processed_paths[0]
            )
            if not np.all(include_frames == self.include_frames):
                raise ValueError(
                    f"the include_frames is changed. "
                    f"please delete the processed folder and rerun {self.processed_paths[0]}"
                )
            self.fixed_fields['dataset_idx'] = self.fixed_fields.get('dataset_idx', 0) * 0 + self.dataset_idx

    def len(self):
        if self.data is None:
            return 0
        return self.data.num_graphs

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pth", "params.yaml"]

    def get_data(
        self,
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], List[AtomicData]]:
        """Get the data --- called from ``process()``.

        Note that parameters for graph construction such as ``pbc`` and ``r_max`` should be included here as (likely, but not necessarily, fixed) fields.

        Returns:
        A two-tuple of:
            fields: dict
                mapping a field name ('pos', 'cell') to a list-like sequence of tensor-like objects giving that field's value for each example.
            fixed_fields: dict
                mapping field names to their constant values for every example in the dataset.
        Or:
            data_list: List[AtomicData]
        """
        raise NotImplementedError

    def download(self):
        if (not hasattr(self, "url")) or (self.url is None):
            # Don't download, assume present. Later could have FileNotFound if the files don't actually exist
            pass
        else:
            download_path = download_url(self.url, self.raw_dir)
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.raw_dir)

    def process(self):
        data = self.get_data()
        if len(data) == 1:

            # It's a data list
            data_list = data[0]
            if not (self.include_frames is None or data[0] is None):
                data_list = [data_list[i] for i in self.include_frames]
            assert all(isinstance(e, AtomicData) for e in data_list)
            assert all(AtomicDataDict.BATCH_KEY not in e for e in data_list)

            fields, fixed_fields = {}, {}

            # take the force_fixed_keys away from the fields
            for key in self.force_fixed_keys:
                if key in data_list[0]:
                    fixed_fields[key] = data_list[0][key]

            fixed_fields.update(self.extra_fixed_fields)

        elif len(data) == 2:

            # It's fields and fixed_fields
            # Get our data
            fields, fixed_fields = data

            fixed_fields.update(self.extra_fixed_fields)

            # check keys
            all_keys = set(fields.keys()).union(fixed_fields.keys())
            assert len(all_keys) == len(fields) + len(
                fixed_fields
            ), "No overlap in keys between data and fixed fields allowed!"
            assert AtomicDataDict.BATCH_KEY not in all_keys
            # Check bad key combinations, but don't require that this be a graph yet.
            AtomicDataDict.validate_keys(all_keys, graph_required=False)

            # take the force_fixed_keys away from the fields
            for key in self.force_fixed_keys:
                if key in fields:
                    fixed_fields[key] = fields.pop(key)[0]
            
            index_fields = {}
            for key in self.force_index_keys:
                if key in fields:
                    index_fields[key] = fields.pop(key)

            # check dimensionality
            num_examples = set([len(a) for a in fields.values()])
            if not len(num_examples) == 1:
                raise ValueError(
                    f"This dataset is invalid: expected all fields to have same length (same number of examples), but they had shapes { {f: v.shape for f, v in fields.items() } }"
                )
            num_examples = next(iter(num_examples))

            include_frames = self.include_frames
            if include_frames is None:
                include_frames = range(num_examples)

            # Make AtomicData from it:
            if AtomicDataDict.EDGE_INDEX_KEY in all_keys:
                # This is already a graph, just build it
                constructor = AtomicData
            else:
                # do neighborlist from points
                constructor = AtomicData.from_points
                assert "r_max" in all_keys
                assert AtomicDataDict.POSITIONS_KEY in all_keys

            data_list = [
                constructor(**{**{f: v[i] for f, v in fields.items()}, **fixed_fields, **index_fields})
                for i in include_frames
            ]

        else:
            raise ValueError("Invalid return from `self.get_data()`")

        # Batch it for efficient saving
        # This limits an AtomicInMemoryDataset to a maximum of LONG_MAX atoms _overall_, but that is a very big number and any dataset that large is probably not "InMemory" anyway
        data = Batch.from_data_list(data_list, exclude_keys=fixed_fields.keys())
        del data_list
        del fields

        # type conversion
        _process_dict(fixed_fields, ignore_fields=["r_max"])

        total_MBs = sum(item.numel() * item.element_size() for _, item in data) / (
            1024 * 1024
        )
        logging.info(
            f"Loaded data: {data}\n    processed data size: ~{total_MBs:.2f} MB"
        )
        del total_MBs

        # use atomic writes to avoid race conditions between
        # different trainings that use the same dataset
        # since those separate trainings should all produce the same results,
        # it doesn't matter if they overwrite each others cached'
        # datasets. It only matters that they don't simultaneously try
        # to write the _same_ file, corrupting it.
        with atomic_write(self.processed_paths[0], binary=True) as f:
            torch.save((data, fixed_fields, self.include_frames), f)
        with atomic_write(self.processed_paths[1], binary=False) as f:
            yaml.dump(self._get_parameters(), f)

        logging.info("Cached processed data to disk")

        self.data = data
        self.fixed_fields = fixed_fields

    def get(self, idx):
        out = self.data.get_example(idx)
        # Add back fixed fields
        for f, v in self.fixed_fields.items():
            out[f] = v
        return out


class TrajDataset(AtomicInMemoryDataset):
    """Load data from a structure file [top, pdb, gro, psf, tpr] + optional trajectory files [trr, xtc].

    To avoid loading unneeded data, keys are ignored by default unless they are in ``key_mapping``, ``include_keys``,
    ``npz_fixed_fields_keys`` or ``extra_fixed_fields``.

    Args:
        key_mapping (Dict[str, str]): mapping of npz keys to ``AtomicData`` keys. Optional
        include_keys (list): the attributes to be processed and stored. Optional
        fixed_field_keys: the attributes that only have one instance but apply to all frames. Optional
            Note that the mapped keys (as determined by the _values_ in ``key_mapping``) should be used in
            ``fixed_field_keys``, not the original npz keys from before mapping. If an npz key is not
            present in ``key_mapping``, it is mapped to itself, and this point is not relevant.

    """

    def __init__(
        self,
        root: str,
        dataset_idx: int,
        structure_filename: str,
        traj_filenames: List[str] = [],
        extra_kwargs: Dict[str, Any] = {},
        selection: str = 'name CA',
        fixed_field_keys: List[str] = [AtomicDataDict.ATOM_TYPE_KEY, AtomicDataDict.PBC_KEY],
        atom_name2type_mapping: Dict[str, int] = {},
        url: Optional[str] = None,
        force_fixed_keys: List[str] = [AtomicDataDict.CELL_KEY, AtomicDataDict.PBC_KEY],
        extra_fixed_fields: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
    ):
        self.structure_filename = structure_filename
        self.traj_filenames = traj_filenames
        self.extra_kwargs = extra_kwargs
        self.selection = selection
        self.atom_name2type_mapping = atom_name2type_mapping

        self.fixed_field_keys = fixed_field_keys

        super().__init__(
            dataset_idx=dataset_idx,
            url=url,
            root=root,
            force_fixed_keys=force_fixed_keys,
            extra_fixed_fields=extra_fixed_fields,
            include_frames=include_frames,
        )

    @property
    def raw_dir(self):
        return dirname(abspath(self.structure_filename))
    
    @property
    def raw_file_names(self):
        return [basename(self.structure_filename)]

    def get_data(self):

        u = mda.Universe(self.structure_filename, *self.traj_filenames, **self.extra_kwargs)
        sel = u.select_atoms(self.selection)

        atom_names = []
        atom_types =[]

        for atom in sel.atoms:
            atom_name = atom.resname + '_' + atom.name
            atom_names.append(atom_name)
            atom_type = self.atom_name2type_mapping.get(atom_name, None)
            if atom_type is None:
                atom_type = len(self.atom_name2type_mapping)
                self.atom_name2type_mapping[atom_name] = atom_type
            atom_types.append(atom_type)

        atom_names = np.array(atom_names)
        atom_types = np.array(atom_types)

        frame_limit = 10
        atom_positions = []
        cell_sizes = []
        traj = u.trajectory if frame_limit is None else u.trajectory[:frame_limit]
        for ts in traj:
            atom_positions.append(sel.positions)
            if ts.dimensions is not None:
                cell_sizes.append(Cell.fromcellpar(ts.dimensions)[:])


        atom_positions = np.stack(atom_positions, axis=0)
        cell = np.stack(cell_sizes, axis=0) if len(cell_sizes) > 0 else np.zeros((3, 3), dtype=np.float32)

        mapped = {
            AtomicDataDict.POSITIONS_KEY: atom_positions,
            AtomicDataDict.ATOM_TYPE_KEY: atom_types,
            AtomicDataDict.CELL_KEY: cell,
            AtomicDataDict.PBC_KEY: np.array((False, False, False))
        }

        fields = {k: fix_batch_dim(v) for k, v in mapped.items() if k not in self.fixed_field_keys}
        # note that we don't deal with extra_fixed_fields here; AtomicInMemoryDataset does that.
        fixed_fields = {
            k: v for k, v in mapped.items() if k in self.fixed_field_keys
        }
        fixed_fields[AtomicDataDict.DATASET_INDEX_KEY] = np.array(self.dataset_idx)

        for key in mapped.keys():
            if key in fields and np.issubdtype(fields[key].dtype, np.integer):
                fields[key] = fields[key].astype(np.int64)
            if key in fixed_fields and np.issubdtype(fixed_fields[key].dtype, np.integer):
                fixed_fields[key] = fixed_fields[key].astype(np.int64)

        return fields, fixed_fields

def fix_batch_dim(arr):
    if len(arr.shape) == 0:
        return arr.reshape(1)
    return arr