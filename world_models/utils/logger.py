from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir):
        date = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        self.logdir = f"{logdir}/{date}"
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.scalars = {}
        self.images = {}
        self.videos = {}
        self.fps = {}

    def add_scalar(self, key, value):
        self.scalars[key] = value

    def add_metrics(self, metrics):
        self.scalars.update(metrics)

    def add_image(self, key, image):
        self.images[key] = image

    def add_video(self, key, video, fps=20):
        # shape should be (N, T, C, H, W)
        self.videos[key] = video
        self.fps[key] = fps

    def write(self, timestep):
        for key, value in self.scalars.items():
            self.writer.add_scalar(key, np.array(value), timestep)

        for key, img in self.images.items():
            if img.dim == 3:
                self.writer.add_image(key, img, timestep)
            else:
                self.writer.add_images(key, img, timestep)

        for key, vid in self.videos.items():
            self.writer.add_video(key, vid, timestep, fps=self.fps[key])

        self.writer.flush()
        self.scalars = {}
        self.images = {}
        self.videos = {}
        self.fps = {}
