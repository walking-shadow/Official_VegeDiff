import os.path as osp
import io
import cv2
import random
import numpy as np
import imageio

from collections import OrderedDict
from cv2 import (CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
                 CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)
from sgm.utils.path_util import (check_file_exist, mkdir_or_exist, scandir,
                            track_progress)
from sgm.utils.ceph_util import PetrelBackend
from typing import Any, Optional, Tuple, Union, Dict, List


class Cache:

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoReader:
    """Video class with similar usage to a list object.

    This video wrapper class provides convenient apis to access frames.
    There exists an issue of OpenCV's VideoCapture class that jumping to a
    certain frame may be inaccurate. It is fixed in this class by checking
    the position after jumping each time.
    Cache is used when decoding videos. So if the same frame is visited for
    the second time, there is no need to decode again if it is stored in the
    cache.

    Examples:
        >>> import mmcv
        >>> v = mmcv.VideoReader('sample.mp4')
        >>> len(v)  # get the total frame number with `len()`
        120
        >>> for img in v:  # v is iterable
        >>>     mmcv.imshow(img)
        >>> v[5]  # get the 6th frame
    """

    def __init__(self, filename, cache_capacity=10):
        # Check whether the video path is a url
        if not filename.startswith(('https://', 'http://')):
            check_file_exist(filename, 'Video file not found: ' + filename)
        self._vcap = cv2.VideoCapture(filename)
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                f'"frame_id" must be between 0 and {self._frame_cnt - 1}')
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
            return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:06d}.jpg',
                   start=0,
                   max_num=0,
                   show_progress=True):
        """Convert a video to frame images.

        Args:
            frame_dir (str): Output directory to store all the frame images.
            file_start (int): Filenames will start from the specified number.
            filename_tmpl (str): Filename template with the index as the
                placeholder.
            start (int): The starting frame index.
            max_num (int): Maximum number of frames to be written.
            show_progress (bool): Whether to show a progress bar.
        """
        mkdir_or_exist(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        def write_frame(file_idx):
            img = self.read()
            if img is None:
                return
            filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
            cv2.imwrite(filename, img)

        if show_progress:
            track_progress(write_frame, range(file_start,
                                              file_start + task_num))
        else:
            for i in range(task_num):
                write_frame(file_start + i)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_frame(i)
                for i in range(*index.indices(self.frame_cnt))
            ]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


def frames2video(frame_dir: str,
                 video_file: str,
                 fps: float = 30,
                 fourcc: str = 'XVID',
                 filename_tmpl: str = '{:06d}.jpg',
                 start: int = 0,
                 end: int = 0,
                 show_progress: bool = True) -> None:
    """Read the frame images from a directory and join them as a video.

    Args:
        frame_dir (str): The directory containing video frames.
        video_file (str): Output filename.
        fps (float): FPS of the output video.
        fourcc (str): Fourcc of the output video, this should be compatible
            with the output file type.
        filename_tmpl (str): Filename template with the index as the variable.
        start (int): Starting frame index.
        end (int): Ending frame index.
        show_progress (bool): Whether to show a progress bar.
    """
    if end == 0:
        ext = filename_tmpl.split('.')[-1]
        end = len([name for name in scandir(frame_dir, ext)])
    first_file = osp.join(frame_dir, filename_tmpl.format(start))
    check_file_exist(first_file, 'The start frame not found: ' + first_file)
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    resolution = (width, height)
    vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps,
                              resolution)

    def write_frame(file_idx):
        filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
        img = cv2.imread(filename)
        vwriter.write(img)

    if show_progress:
        track_progress(write_frame, range(start, end))
    else:
        for i in range(start, end):
            write_frame(i)
    vwriter.release()
    

class DecordInit():
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - filename

    Added Keys:

        - video_reader
        - total_frames
        - fps

    Args:
        io_backend (str): io backend where frames are store.
            Defaults to ``'disk'``.
        num_threads (int): Number of thread to decode the video. Defaults to 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 file_client,
                 num_threads: int = 1,
                 **kwargs) -> None:
        # self.io_backend = io_backend
        self.file_client = file_client
        self.num_threads = num_threads
        self.kwargs = kwargs

    def _get_video_reader(self, filename: str) -> object:
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        # if self.file_client is None:
            # self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        return container

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord initialization.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = self._get_video_reader(results['filename'])
        results['total_frames'] = len(container)

        results['video_reader'] = container
        results['avg_fps'] = container.get_avg_fps()
        return results


    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


class DecordDecode():
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - video_reader
        - frame_inds

    Added Keys:

        - imgs
        - original_shape
        - img_shape

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets.
            Defaults to ``'accurate'``.
    """

    def __init__(self, mode: str = 'accurate') -> None:
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container: object,
                            frame_inds: np.ndarray) -> List[np.ndarray]:
        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())
        return imgs

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord decoding.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results


    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str

class PyAVInit():
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, file_client, **kwargs):
        self.file_client = file_client
        self.kwargs = kwargs

    def transform(self, results):
        """Perform the PyAV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        # if self.file_client is None:
            # self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj)

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend={self.io_backend})'
        return repr_str


class PyAVDecode():
    """Using PyAV to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            the nearest key frames, which may be duplicated and inaccurate,
            and more suitable for large scene-based video datasets.
            Default: 'accurate'.
    """

    def __init__(self, multi_thread=False, mode='accurate'):
        self.multi_thread = multi_thread
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    @staticmethod
    def frame_generator(container, stream):
        """Frame generator for PyAV."""
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def transform(self, results):
        """Perform the PyAV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        if self.mode == 'accurate':
            # set max indice to make early stop
            max_inds = max(results['frame_inds'])
            i = 0
            for frame in container.decode(video=0):
                if i > max_inds + 1:
                    break
                imgs.append(frame.to_rgb().to_ndarray())
                i += 1

            # the available frame in pyav may be less than its length,
            # which may raise error
            results['imgs'] = [
                imgs[i % len(imgs)] for i in results['frame_inds']
            ]
        elif self.mode == 'efficient':
            for frame in container.decode(video=0):
                backup_frame = frame
                break
            stream = container.streams.video[0]
            for idx in results['frame_inds']:
                pts_scale = stream.average_rate * stream.time_base
                frame_pts = int(idx / pts_scale)
                container.seek(
                    frame_pts, any_frame=False, backward=True, stream=stream)
                frame = self.frame_generator(container, stream)
                if frame is not None:
                    imgs.append(frame)
                    backup_frame = frame
                else:
                    imgs.append(backup_frame)
            results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['video_reader'] = None
        del container

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread}, mode={self.mode})'
        return repr_str

class PetrelVideoReader():
    """
    file_client: pretrel_client
    num_threads: num_threads for decord
    """
    def __init__(self, file_client):
        self.file_client = file_client
        # self.num_threads = num_threads
        # self.videodecoder = DecordDecode()
        # self.videoreader = DecordInit(self.file_client, self.num_threads)
        self.videoreader = PyAVInit(self.file_client)
        self.videodecoder = PyAVDecode()
        
    def sample_whole_video(self, filename):
        results = dict()
        results['filename'] = filename
        results = self.videoreader.transform(results)
        results['frame_inds'] = np.arange(results["total_frames"])
        results = self.videodecoder.transform(results)
        return results
    
    def sample_clip(self, clip_length, clip_FPS_reate, filename):
        '''
        standard FPS:30
        clip_length: 20 (frames)
        clip_FPS_reate (or frame interval): 1 (1: fps=30, 2: fps=15 ; 
            if your video fps=60, choose clip_FPS_reate=2, make your video fps=30)
        '''
        results = dict()
        results['filename'] = filename
        results = self.videoreader.transform(results)
        # print(results['filename'], results["total_frames"]-1-(clip_length-1)*clip_FPS_reate,results["total_frames"],(clip_length-1),clip_FPS_reate)
        start_random_index = random.randint(0, results["total_frames"]-1-(clip_length-1)*clip_FPS_reate)
        end_random_index = start_random_index + (clip_length-1)*clip_FPS_reate
        results['frame_inds'] = np.array([i for i in range(start_random_index, end_random_index+1, clip_FPS_reate)])
        results = self.videodecoder.transform(results)
        results['clip_length'] = clip_length
        results['clip_FPS_reate'] = clip_FPS_reate
        return results
    
class PetrelVideo():
    """
    file_client: pretrel_client
    num_threads: num_threads for decord
    """
    def __init__(self, file_client):
        self.file_client = file_client
        
    def get_clip_from_id(self, video_frames, frame_inds):
        return video_frames[frame_inds]
        
    def sample_whole_video(self, filename):
        results = self.file_client.get_video(filepath=filename)
        results['frame_inds'] = np.arange(results["total_frames"])
        results["imgs"] = self.get_clip_from_id(video_frames=results["video"], frame_inds=results['frame_inds'])
        return results

    def sample_clip(self, clip_length, clip_FPS_reate, filename):
        '''
        standard FPS:30
        clip_length: 20 (frames)
        clip_FPS_reate (or frame interval): 1 (1: fps=30, 2: fps=15 ; 
            if your video fps=60, choose clip_FPS_reate=2, make your video fps=30)
        '''
        results = self.file_client.get_video(filepath=filename)
        # print(results['filename'], results["total_frames"]-1-(clip_length-1)*clip_FPS_reate,results["total_frames"],(clip_length-1),clip_FPS_reate)
        if results["total_frames"]-1-(clip_length-1)*clip_FPS_reate >= 0:
            start_random_index = random.randint(0, results["total_frames"]-1-(clip_length-1)*clip_FPS_reate)
            end_random_index = start_random_index + (clip_length-1)*clip_FPS_reate
            results['frame_inds'] = np.array([i for i in range(start_random_index, end_random_index+1, clip_FPS_reate)])
            results["imgs"] = self.get_clip_from_id(video_frames=results["video"], frame_inds=results['frame_inds'])
            results['clip_length'] = clip_length
            results['clip_FPS_reate'] = clip_FPS_reate
        else: # Padding Last Frame
            results['frame_inds'] = np.arange(results["total_frames"])
            results["imgs"] = self.get_clip_from_id(video_frames=results["video"], frame_inds=results['frame_inds'])
            padding_length = clip_length - results["total_frames"]
            padding_frame = results["imgs"][-1]
            expanded_frame = np.expand_dims(padding_frame, axis=0)
            repeated_frame = np.repeat(expanded_frame, padding_length, axis=0)
            results["imgs"] = np.concatenate((results["imgs"], repeated_frame), axis=0)
            results['clip_length'] = clip_length
            results['clip_FPS_reate'] = clip_FPS_reate
        return results
    
def numpy_array_to_video(numpy_array, video_out_path, fps=8): #TCHW -> THWC
    numpy_array = np.transpose(numpy_array, (0, 2, 3, 1)) #THWC
    with imageio.get_writer(video_out_path, fps=fps) as video:
        for image in numpy_array:
            video.append_data(image)
            
            
class PetrelDecordReader():
    """
    file_client: pretrel_client
    num_threads: num_threads for decord
    """
    def __init__(self, file_client):
        self.file_client = file_client
        self.videodecoder = DecordDecode()
        self.videoreader = DecordInit(self.file_client)
        
    def sample_whole_video(self, filename):
        results = dict()
        results['filename'] = filename
        results = self.videoreader.transform(results)
        results['frame_inds'] = np.arange(results["total_frames"])
        results = self.videodecoder.transform(results)

        return results
    
    def sample_clip(self, clip_length, clip_FPS_reate, filename):
        '''
        standard FPS:30
        clip_length: 20 (frames)
        clip_FPS_reate (or frame interval): 1 (1: fps=30, 2: fps=15 ; 
            if your video fps=60, choose clip_FPS_reate=2, make your video fps=30)
        '''
        results = dict()
        results['filename'] = filename
        results = self.videoreader.transform(results)
        results["total_frames"]=2
        if results["total_frames"]-1-(clip_length-1)*clip_FPS_reate >= 0:
            start_random_index = random.randint(0, results["total_frames"]-1-(clip_length-1)*clip_FPS_reate)
            end_random_index = start_random_index + (clip_length-1)*clip_FPS_reate
            results['frame_inds'] = np.array([i for i in range(start_random_index, end_random_index+1, clip_FPS_reate)])
            results = self.videodecoder.transform(results)
            results['clip_length'] = clip_length
            results['clip_FPS_reate'] = clip_FPS_reate
        else:
            results['frame_inds'] = np.arange(results["total_frames"])
            results = self.videodecoder.transform(results)
            padding_length = clip_length - results["total_frames"]
            padding_frame = results["imgs"][-1]
            expanded_frame = np.expand_dims(padding_frame, axis=0)
            repeated_frame = np.repeat(expanded_frame, padding_length, axis=0)
            results["imgs"] = np.concatenate((results["imgs"], repeated_frame), axis=0)
            results['clip_length'] = clip_length
            results['clip_FPS_reate'] = clip_FPS_reate
        return results