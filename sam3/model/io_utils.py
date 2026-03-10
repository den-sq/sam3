# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import abc
import os
import re
import time
from threading import Lock, Thread

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sam3.logger import get_logger

logger = get_logger(__name__)

IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
RANK = int(os.getenv("RANK", "0"))

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


class VideoLoader(abc.ABC):
    def __init__(
        self,
        num_frames,
        image_size,
        img_mean,
        img_std,
        device="cpu",
        video_height=None,
        video_width=None,
    ):
        self.num_frames = num_frames
        self.image_size = image_size
        # Normalize mean/std to tensors so frame normalization arithmetic is valid.
        self.img_mean = torch.as_tensor(img_mean, dtype=torch.float16).reshape(-1, 1, 1)
        self.img_std = torch.as_tensor(img_std, dtype=torch.float16).reshape(-1, 1, 1)
        self.device = device
        # Mock Tensor properties for compatibility
        self.shape = (num_frames, 3, image_size, image_size)
        self.ndim = 4
        self.video_height = video_height
        self.video_width = video_width

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def cleanup(self, current_idx, window_size=5, reverse=False):
        """Unload frames that are no longer needed based on the current index and window."""
        pass

    def __len__(self):
        return self.num_frames

    def to(self, *args, **kwargs):
        # Mocking .to() to be compatible with typical tensor operations, though strictly no-op here
        return self

    def close(self):
        """Release loader resources. Subclasses can override."""
        return None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors should never raise.
            pass


class SyncVideoLoader(VideoLoader):
    def __init__(
        self,
        resource_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        is_video_file=False,
    ):
        num_frames = 0
        video_height = None
        video_width = None
        self.is_video_file = is_video_file
        self.resource_path = resource_path
        self.frame_names = []
        self.cap = None

        if is_video_file:
            import cv2
            self.cap = cv2.VideoCapture(resource_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {resource_path}")
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            # Image folder
            self.frame_names = [
                p for p in os.listdir(resource_path) if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
            ]
            try:
                self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            except ValueError:
                self.frame_names.sort()
            num_frames = len(self.frame_names)
            if num_frames == 0:
                raise RuntimeError(f"no images found in {resource_path}")
            # Load first frame to get dims
            _, video_height, video_width = _load_img_as_tensor(
                os.path.join(resource_path, self.frame_names[0]), image_size
            )
        if num_frames <= 0:
            raise RuntimeError(f"Could not determine a positive frame count for {resource_path}")

        device = "cpu" if offload_video_to_cpu else "cuda"
        super().__init__(
            num_frames,
            image_size,
            img_mean,
            img_std,
            device,
            video_height=video_height,
            video_width=video_width,
        )
        self.cache = {}

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of bounds for {self.num_frames} frames")
        if idx in self.cache:
            return self.cache[idx]

        if self.is_video_file:
            import cv2
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Failed to load frame {idx} from video {self.resource_path}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(
                frame_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC
            )
            img_tensor = torch.from_numpy(frame_resized.astype(np.float32)).permute(2, 0, 1) / 255.0
            # Normalize
            img_tensor = (img_tensor.to(torch.float16) - self.img_mean) / self.img_std
        else:
            img_path = os.path.join(self.resource_path, self.frame_names[idx])
            img_tensor, _, _ = _load_img_as_tensor(img_path, self.image_size)
            img_tensor = (img_tensor.to(torch.float16) - self.img_mean) / self.img_std

        if self.device != "cpu":
            img_tensor = img_tensor.cuda()

        self.cache[idx] = img_tensor
        return img_tensor

    def cleanup(self, current_idx, window_size=5, reverse=False):
        keys_to_remove = []
        for idx in list(self.cache.keys()):
            if not reverse:
                # Forward: remove frames that are too far in the past
                if idx < current_idx - window_size:
                    keys_to_remove.append(idx)
            else:
                # Reverse: remove frames that are too far in the future
                if idx > current_idx + window_size:
                    keys_to_remove.append(idx)

        for k in keys_to_remove:
            del self.cache[k]

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cache.clear()


class AsyncVideoLoader(VideoLoader):
    def __init__(
        self,
        resource_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        buffer_size=10,
        is_video_file=False,
    ):
        # Initialization similar to Sync, but setup async thread
        num_frames = 0
        video_height = None
        video_width = None
        self.is_video_file = is_video_file
        self.resource_path = resource_path
        self.buffer_size = buffer_size
        self.frame_names = []

        if is_video_file:
            import cv2
            cap = cv2.VideoCapture(resource_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        else:
            self.frame_names = [
                p for p in os.listdir(resource_path) if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
            ]
            try:
                self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            except ValueError:
                self.frame_names.sort()
            num_frames = len(self.frame_names)
            if num_frames == 0:
                raise RuntimeError(f"no images found in {resource_path}")
            _, video_height, video_width = _load_img_as_tensor(
                os.path.join(resource_path, self.frame_names[0]), image_size
            )
        if num_frames <= 0:
            raise RuntimeError(f"Could not determine a positive frame count for {resource_path}")

        device = "cpu" if offload_video_to_cpu else "cuda"
        super().__init__(
            num_frames,
            image_size,
            img_mean,
            img_std,
            device,
            video_height=video_height,
            video_width=video_width,
        )

        self.cache = {}
        self.lock = Lock()
        self.current_idx = 0
        self.target_idx = 0  # Where the async loader tries to reach
        self.stop_event = False

        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        import cv2
        cap = None
        if self.is_video_file:
            cap = cv2.VideoCapture(self.resource_path)

        while not self.stop_event:
            # Determine range to load: [current_idx, current_idx + buffer_size]
            with self.lock:
                start_load = self.target_idx
                end_load = min(self.target_idx + self.buffer_size, self.num_frames)

            if start_load >= end_load:
                time.sleep(0.01)
                continue

            for idx in range(start_load, end_load):
                if self.stop_event:
                    break

                # Skip if already in cache
                with self.lock:
                    if idx in self.cache:
                        continue

                # Load logic
                try:
                    if self.is_video_file:
                        # Optimization: only seek if not sequential
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_resized = cv2.resize(
                                frame_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC
                            )
                            img_tensor = torch.from_numpy(frame_resized.astype(np.float32)).permute(2, 0, 1) / 255.0
                            img_tensor = (img_tensor.to(torch.float16) - self.img_mean) / self.img_std
                        else:
                            img_tensor = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float16)
                    else:
                        img_path = os.path.join(self.resource_path, self.frame_names[idx])
                        img_tensor, _, _ = _load_img_as_tensor(img_path, self.image_size)
                        img_tensor = (img_tensor.to(torch.float16) - self.img_mean) / self.img_std

                    if self.device != "cpu":
                        img_tensor = img_tensor.cuda()

                    with self.lock:
                        self.cache[idx] = img_tensor
                except Exception as e:
                    logger.error(f"Async loading error frame {idx}: {e}")

            time.sleep(0.01)

        if cap:
            cap.release()

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of bounds for {self.num_frames} frames")
        # Signal worker to prioritize this area
        with self.lock:
            self.target_idx = idx

        # Wait for frame
        start_time = time.time()
        while True:
            with self.lock:
                if idx in self.cache:
                    return self.cache[idx]
            if time.time() - start_time > 10.0:
                raise TimeoutError(f"Timed out loading frame {idx}")
            time.sleep(0.005)

    def cleanup(self, current_idx, window_size=5, reverse=False):
        with self.lock:
            keys_to_remove = []
            for idx in list(self.cache.keys()):
                if not reverse:
                    if idx < current_idx - window_size:
                        keys_to_remove.append(idx)
                else:
                    if idx > current_idx + window_size:
                        keys_to_remove.append(idx)
            for k in keys_to_remove:
                del self.cache[k]

    def close(self):
        self.stop_event = True
        thread = getattr(self, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self.thread = None
        with self.lock:
            self.cache.clear()


def load_resource_as_video_frames(
    resource_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    video_loader_type="cv2",
    async_buffer_size=10,
):
    """
    Returns a VideoLoader object instead of a full tensor.
    """
    if isinstance(resource_path, str) and resource_path.startswith("<load-dummy-video"):
        if video_loader_type != "cv2":
            raise RuntimeError("video_loader_type must be 'cv2' for dummy video loading")
        # Pattern: <load-dummy-video-N> where N is an integer.
        match = re.match(r"<load-dummy-video-(\d+)>", resource_path)
        num_frames = int(match.group(1)) if match else 60
        return load_dummy_video(image_size, offload_video_to_cpu, num_frames=num_frames)

    # Handle single image or list of images case (legacy behavior for image inference)
    if isinstance(resource_path, list) or (
        isinstance(resource_path, str) and os.path.splitext(resource_path)[-1].lower() in IMAGE_EXTS
    ):
        # Fallback to loading everything for single images/lists as it's efficient enough
        if isinstance(resource_path, str):
            return load_image_as_single_frame_video(
                resource_path, image_size, offload_video_to_cpu, img_mean, img_std
            )
        else:
            # List of PIL images
            if len(resource_path) == 0:
                raise RuntimeError("resource_path list is empty")
            img_mean_t = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
            img_std_t = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
            orig_height, orig_width = resource_path[0].size
            orig_height, orig_width = orig_width, orig_height
            images = []
            for img_pil in resource_path:
                img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
                img = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float16) / 255.0
                img = (img - img_mean_t) / img_std_t
                images.append(img)
            images = torch.stack(images)
            if not offload_video_to_cpu:
                images = images.cuda()
            return images, orig_height, orig_width

    # Check resource type
    is_video_file = isinstance(resource_path, str) and os.path.splitext(resource_path)[-1].lower() in VIDEO_EXTS
    is_image_folder = isinstance(resource_path, str) and os.path.isdir(resource_path)

    if not (is_video_file or is_image_folder):
        raise NotImplementedError("Only video files and image folders are supported for VideoLoader")
    if video_loader_type != "cv2":
        raise RuntimeError("video_loader_type must be 'cv2' for VideoLoader")

    if async_loading_frames:
        loader = AsyncVideoLoader(
            resource_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            buffer_size=async_buffer_size,
            is_video_file=is_video_file
        )
    else:
        loader = SyncVideoLoader(
            resource_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            is_video_file=is_video_file
        )

    return loader, loader.video_height, loader.video_width


def load_image_as_single_frame_video(
    image_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
):
    """Load an image as a single-frame video."""
    images, image_height, image_width = _load_img_as_tensor(image_path, image_size)
    images = images.unsqueeze(0).half()

    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, image_height, image_width


def load_dummy_video(image_size, offload_video_to_cpu, num_frames=60):
    """Load a dummy video for warm-up/compilation purposes."""
    video_height, video_width = 480, 640
    images = torch.randn(num_frames, 3, image_size, image_size, dtype=torch.float16)
    if not offload_video_to_cpu:
        images = images.cuda()
    return images, video_height, video_width


def _load_img_as_tensor(img_path, image_size):
    """Load and resize an image and convert it into a PyTorch tensor."""
    img = Image.open(img_path).convert("RGB")
    orig_width, orig_height = img.width, img.height
    img = TF.resize(img, size=(image_size, image_size))
    img = TF.to_tensor(img)
    return img, orig_height, orig_width
