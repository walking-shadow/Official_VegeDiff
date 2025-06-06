import os
import os.path as osp
import re
import json
import pickle
import warnings
import cv2
import numpy as np
import torch
import torchvision

from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Tuple, Union




def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.

    Args:
        method (str): The method name to check.
        obj (object): The object to check.
    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


class PetrelBackend:
    """Petrel storage backend - simple version"""

    def __init__(self,
                 path_mapping: Optional[dict] = None,
                 enable_mc: bool = False,
                 conf_path: Optional[str] = None):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(conf_path=conf_path, enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str or Path): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def _replace_prefix(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        return filepath.replace('petrel://', 's3://')

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        if not has_method(self._client, 'isdir'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `isdir` method, please use a higher version or dev'
                ' branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        if not has_method(self._client, 'contains'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `contains` method, please use a higher version or '
                'dev branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath)

    def join_path(
        self,
        filepath: Union[str, Path],
        *filepaths: Union[str, Path],
    ) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result after concatenation.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.join_path(filepath, 'another/path')
            'petrel://path/of/file/another/path'
            >>> backend.join_path(filepath, '/another/path')
            'petrel://path/of/file/another/path'
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith('/'):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._format_path(self._map_path(path))
            formatted_paths.append(formatted_path.lstrip('/'))

        return '/'.join(formatted_paths)

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Return bytes read from filepath.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        value = self._client.Get(filepath)
        return value

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.put(b'hello world', filepath)
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.put(filepath, obj)

    def get_text(self, filepath, warning=False) -> str:
        try:
            value = self._client.Get(filepath)
        except Exception:
            if warning:
                warnings.warn(f"Failed to get text from {filepath}")
                value = None
            else:
                raise Exception(f"Failed to get text from {filepath}")
        return str(value, encoding="utf-8")

    def get_uint16_png(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.frombuffer(self._client.get(filepath), np.uint8)
            value = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        except Exception:
            if warning:
                warnings.warn(f"Failed to get uint16_png from {filepath}")
                value = None
            else:
                raise Exception(f"Failed to get uint16_png from {filepath}")
        return value

    def get_uint8_jpg(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.frombuffer(self._client.get(filepath), np.uint8)
            value = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        except Exception:
            if warning:
                warnings.warn(f"Failed to get uint8_jpg from {filepath}")
                value = None
            else:
                raise Exception(f"Failed to get uint8_jpg from {filepath}")
        return value

    def get_npz(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = np.loads(value)
        except Exception:
            if warning:
                warnings.warn(f"Failed to get npz from {filepath}")
                value = None
            else:
                raise Exception(f"Failed to get npz from {filepath}")
        return value

    def get_numpy_txt(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.loadtxt(StringIO(self.get_text(filepath)))
        except Exception:
            if warning:
                warnings.warn(f"Failed to get numpy_txt from {filepath}")
                value = None
            else:
                raise Exception(f"Failed to get numpy_txt from {filepath}")
        return value

    def get_json(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = json.loads(value)
        except Exception:
            if warning:
                warnings.warn(f"Failed to get json from {filepath}")
                value = None
            else:
                raise Exception(f"Failed to get json from {filepath}")
        return value

    # discard
    def get_video(self, filepath, warning=True):
        '''
        input: filepath
        output: {
            video: ndarray, format (TCHW)
            meta: metadata {"video": {"fps", "duration"}}
        }
        '''
        try:
            value = self._client.get(filepath)
            print("start torchvision")
            reader = torchvision.io.VideoReader(src=value, num_threads=1)
            print("start read")
            frames = []
            for frame in reader:
                # print(len(frames))
                if len(frames) > 100:
                    break
                    del(frames[0])
                frames.append(frame['data'])
            print("start stack and numpy")
            value = torch.stack(frames, dim=0).numpy()
            print("start read meta")
            reader_md = reader.get_metadata()
            reader = None
            print("end read meta")
            output = {'video': value, 
                      'filename': filepath, 
                      'total_frames': value.shape[0],
                      'shape': value.shape,
                      'meta': reader_md}
            return  output
        except Exception:
            if warning:
                warnings.warn(f"Failed to get video from {filepath}")
                value = None
            else:
                raise Exception(f"Failed to get video from {filepath}")
        return value

    def put_uint16_png(self, filepath, value) -> None:
        success, img_array = cv2.imencode(".png", value, params=[cv2.CV_16U])
        assert success
        img_bytes = img_array.tobytes()
        self._client.put(filepath, img_bytes)
        # self._client.put(filepath, img_bytes, update_cache=True)

    def put_uint8_jpg(self, filepath, value) -> None:
        success, img_array = cv2.imencode(".jpg", value)
        assert success
        img_bytes = img_array.tobytes()
        self._client.put(filepath, img_bytes)
        # self._client.put(filepath, img_bytes, update_cache=True)

    def put_npz(self, filepath, value) -> None:
        value = pickle.dumps(value)
        self._client.put(filepath, value)
        # self._client.put(filepath, value, update_cache=True)

    def put_json(self, filepath, value) -> None:
        value = json.dumps(value).encode()
        self._client.put(filepath, value)
        # self._client.put(filepath, value, update_cache=True)

    def put_text(self, filepath, value) -> None:
        self._client.put(filepath, bytes(value, encoding="utf-8"))
        # self._client.put(filepath, bytes(value, encoding='utf-8'), update_cache=True)

    def join_path(self, filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        """Concatenate all file paths.

        Args:
            filepath (str or Path): Path to be concatenated.
        Returns:
            str: The result after concatenation.
        """
        # filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith("/"):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_paths.append(path)
        return "/".join(formatted_paths)

    # from mmcv
    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the
                directory. Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = PetrelBackend()
            >>> dir_path = 'petrel://path/of/dir'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path, list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path, list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path, suffix='.txt'):
            ...     print(file_path)
            >>> # list all files and directory recursively
            >>> for file_path in backend.list_dir_or_file(dir_path, recursive=True):
            ...     print(file_path)
        """  # noqa: E501
        if not has_method(self._client, 'list'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `list` method, please use a higher version or dev'
                ' branch instead.')

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        dir_path = self._replace_prefix(dir_path)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith('/'):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root):-1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(next_dir_path, list_dir,
                                                     list_file, suffix,
                                                     recursive)
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root):]
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)

    # from mmcv
    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.
        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        if not (has_method(self._client, "contains") and has_method(self._client, "isdir")):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `contains` and `isdir` methods, please use a higher"
                "version or dev branch instead."
            )

        return self._client.contains(filepath) or self._client.isdir(filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.remove(filepath)
        """
        if not has_method(self._client, 'delete'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `delete` method, please use a higher version or dev '
                'branch instead.')

        if not self.exists(filepath):
            raise FileNotFoundError(f'filepath {filepath} does not exist')

        if self.isdir(filepath):
            raise IsADirectoryError('filepath should be a file')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.delete(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> backend = PetrelBackend()
            >>> dir_path = 'petrel://path/of/dir'
            >>> backend.rmtree(dir_path)
        """
        for path in self.list_dir_or_file(
                dir_path, list_dir=False, recursive=True):
            filepath = self.join_path(dir_path, path)
            self.remove(filepath)

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'path/of/your/dir'
            >>> dst = 'petrel://path/of/dir1'
            >>> backend.copytree_from_local(src, dst)
            'petrel://path/of/dir1'
        """
        dst = self._format_path(self._map_path(dst))
        if self.exists(dst):
            raise FileExistsError('dst should not exist')

        src = str(src)

        for cur_dir, _, files in os.walk(src):
            for f in files:
                src_path = osp.join(cur_dir, f)
                dst_path = self.join_path(dst, src_path.replace(src, ''))
                self.copyfile_from_local(src_path, dst_path)

        return dst
    
    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        src and dst should have the same prefix. If dst specifies a directory,
        the file will be copied into dst using the base filename from src. If
        dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> # dst is a file
            >>> src = 'petrel://path/of/file'
            >>> dst = 'petrel://path/of/file1'
            >>> backend.copyfile(src, dst)
            'petrel://path/of/file1'

            >>> # dst is a directory
            >>> dst = 'petrel://path/of/dir'
            >>> backend.copyfile(src, dst)
            'petrel://path/of/dir/file'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, src.split('/')[-1])

        if src == dst:
            raise SameFileError('src and dst should not be same')

        self.put(self.get(src), dst)
        return dst
    
    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'petrel://path/of/dir'
            >>> dst = 'petrel://path/of/dir1'
            >>> backend.copytree(src, dst)
            'petrel://path/of/dir1'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))

        if self.exists(dst):
            raise FileExistsError('dst should not exist')

        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            src_path = self.join_path(src, path)
            dst_path = self.join_path(dst, path)
            self.put(self.get(src_path), dst_path)

        return dst
