import numpy as np
import torch


class ReplayBuffer:
    # The vanilla replay buffer
    def __init__(
        self,
        buffer_shapes=None,
        dtypes=None,
        buffer_size=1_000_000,
        prioritized=False,
        done_index=-1,
    ):
        self.buffers = []
        self.index = 0
        self.buffer_size = buffer_size
        self.size = 0
        self.done_index = done_index
        self.initialized = False
        self.weights = None

        if buffer_shapes is not None:
            self.init_buffers(buffer_shapes, dtypes)

    def init_buffers(self, buffer_shapes, dtypes=None):
        if dtypes is not None:
            for shape, dtype in zip(buffer_shapes, dtypes):
                self.buffers.append(np.empty((self.buffer_size,) + shape, dtype=dtype))
        else:
            for shape in buffer_shapes:
                self.buffers.append(np.empty((self.buffer_size,) + shape))
        self.temp_buffer = [[] for _ in range(len(self.buffers))]
        self.initialized = True

    def lazy_init(self, sample):
        shapes = []
        dtypes = []
        for s in sample:
            s = np.asarray(s)
            shapes.append(s.shape)
            dtypes.append(s.dtype)
        self.init_buffers(shapes, dtypes)

    def sample_indices(self, batch_size, seq_len=1):
        if self.weights is None:
            if self.size < self.buffer_size:
                if self.size - seq_len < 0:
                    raise ValueError(f"buffer size={self.size} < seq_len={seq_len}")
                indices = np.random.randint(
                    0, min(self.size - seq_len, self.buffer_size), size=batch_size
                )
            else:
                idx = np.random.randint(0, self.buffer_size - seq_len, size=batch_size)
                idx = (self.index + idx) % self.buffer_size
                indices = idx
        indices = (indices[:, None] + np.arange(seq_len)[None, :]) % self.buffer_size
        if seq_len == 1:
            indices = indices[:, 0]
        return indices

    def sample(self, batch_size=256, seq_len=1, indices=None):
        # valid indices go from [0, pos - seq_len) and [pos, buffer_size-seq_len] to prevent the
        # out of order sampling, instead of needing to wait for episode to terminate
        if indices is None:
            indices = self.sample_indices(batch_size, seq_len)
        elif seq_len > 1:
            indices = np.expand_dims(indices, axis=-1) + np.arange(seq_len)
            indices %= self.buffer_size
            indices = indices.transpose()
        sampled = [buffer[indices] for buffer in self.buffers]
        return sampled

    def add_sample(self, sample):
        if not self.initialized:
            self.lazy_init(sample)
        for i, buffer in enumerate(self.buffers):
            buffer[self.index] = sample[i]
        self.index = (self.index + 1) % self.buffer_size
        self.size += 1

    def add_samples(self, data_samples):
        # used mainly for adding full episodes but can also handle small rollouts
        # assumes data_samples is a list of num_steps * dimension
        num_steps = data_samples[0].shape[0]
        indices = (self.index + np.arange(num_steps)) % self.buffer_size
        for i, buffer in enumerate(self.buffers):
            buffer[indices] = data_samples[i]
        self.index = (self.index + num_steps) % self.buffer_size
        self.size += num_steps

    def modify_indices(self, buffer_num, indices, data):
        # might not be needed for this but would be good for prioritized replay or dreamerv3
        raise NotImplementedError

    def modify_weights(self, indices, weights):
        raise NotImplementedError

    def get_state(self):
        state = {
            "index": self.index,
            "total": self.size,
            "weights": self.weights if self.weights is not None else np.array([]),
        }

        for i, buf in enumerate(self.buffers):
            state[f"buffer_{i}"] = buf

        return state

    def save(self, filepath):
        np.savez_compressed(filepath, **self.get_state())

    def set_state(self, state):
        self.index = int(state["index"])
        self.size = int(state["total"])
        self.weights = state["weights"] if state["weights"].size > 0 else None

        for i in range(len(self.buffers)):
            self.buffers[i] = state[f"buffer_{i}"]

    def load(self, filepath):
        with np.load(filepath, allow_pickle=True) as state:
            self.set_state(state)


class TrajectoryReplayBuffer(ReplayBuffer):
    # Serves as both the vanilla replay buffer
    # but can also sample things from within the same trajectory for things like goal-based RL
    def __init__(self, obs_index=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.obs_index = obs_index
        self.indices = np.empty((self.buffer_size, 2), dtype=np.int32)
        self.trajectory_idx = 0
        self.t = 0
        self.trajectory_ends = {}

    def add_sample(self, sample):
        if self.size >= self.buffer_size:
            old_traj_idx = self.indices[self.index, 0]
            if self.trajectory_ends.get(old_traj_idx) == self.index:
                self.trajectory_ends.pop(old_traj_idx, None)

        self.indices[self.index] = [self.trajectory_idx, self.t]

        if sample[self.done_index]:
            self.trajectory_ends[self.trajectory_idx] = self.index
            self.trajectory_idx += 1
            self.t = 0
        else:
            self.t += 1

        super().add_sample(sample)

    def add_samples(self, data_samples):
        num_steps = data_samples[0].shape[0]
        write_indices = (self.index + np.arange(num_steps)) % self.buffer_size

        if self.size >= self.buffer_size:
            old_traj_ids = self.indices[write_indices, 0]
            for i in range(num_steps):
                old_id = old_traj_ids[i]
                if self.trajectory_ends.get(old_id) == write_indices[i]:
                    self.trajectory_ends.pop(old_id, None)

        # pre-allocate the array
        tracking_data = np.zeros((num_steps, 2), dtype=np.int32)

        for i in range(num_steps):
            tracking_data[i] = [self.trajectory_idx, self.t]
            if data_samples[self.done_index][i]:
                self.trajectory_ends[self.trajectory_idx] = write_indices[i]
                self.trajectory_idx += 1
                self.t = 0
            else:
                self.t += 1

        self.indices[write_indices] = tracking_data

        super().add_samples(data_samples)

    def sample_goals(self, indices, discount=0.99):
        # samples a future state, weighted by a discount
        # based on how far they are from the current state
        goal_indices = np.zeros_like(indices)

        for i, start_idx in enumerate(indices):
            target_traj_idx = self.indices[start_idx][0]

            end_idx = self.trajectory_ends.get(target_traj_idx, (self.index - 1) % self.buffer_size)

            # handling circular indice
            if start_idx <= end_idx:
                future_indices = np.arange(start_idx + 1, end_idx + 1)
            else:
                future_indices = np.concatenate(
                    [np.arange(start_idx + 1, self.buffer_size), np.arange(0, end_idx + 1)]
                )

            # edge case of using the last step, in which case the goal is itself
            if len(future_indices) == 0:
                goal_indices[i] = start_idx
                continue

            distances = np.arange(1, len(future_indices) + 1)
            weights = discount**distances
            weights = weights / np.sum(weights)  # Normalize

            goal_indices[i] = np.random.choice(future_indices, p=weights)

        return self.buffers[self.obs_index][goal_indices]

    def sample_with_goals(self, batch_size=256, seq_len=1, discount=0.99):
        indices = self.sample_indices(batch_size, seq_len)
        goal_indices = indices[:, 0] if seq_len > 1 else indices
        data = self.sample(indices=indices)
        goals = self.sample_goals(goal_indices, discount)
        traj_ids = self.indices[goal_indices, 0]
        return data, goals, traj_ids

    def get_state(self):
        state = super().get_state()

        state["indices"] = self.indices
        state["trajectory_idx"] = self.trajectory_idx
        state["t"] = self.t

        state["trajectory_ends"] = np.array([self.trajectory_ends], dtype=object)

        return state

    def set_state(self, state):
        super().set_state(state)

        self.indices = state["indices"]
        self.trajectory_idx = int(state["trajectory_idx"])
        self.t = int(state["t"])

        self.trajectory_ends = state["trajectory_ends"][0]


class PerEnvBuffer:
    def __init__(
        self,
        num_envs,
        buffer_shapes=None,
        dtypes=None,
        buffer_size=1_000_000,
        prioritized=False,
        buffer_class=ReplayBuffer,
    ):
        self.buffers = [
            buffer_class(
                buffer_shapes=buffer_shapes,
                dtypes=dtypes,
                buffer_size=buffer_size // num_envs,
                prioritized=prioritized,
            )
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs
        self._num_items = None
        self.size = 0

    @property
    def num_items(self):
        if self._num_items is None:
            self._num_items = len(self.buffers[0].buffers)
        return self._num_items

    def add_sample(self, sample, idxs=None):
        self.size += self.num_envs
        if idxs is None:
            idxs = list(range(self.num_envs))
        for i in idxs:
            self.buffers[i].add_sample([s[i] for s in sample])

    def add_sample_until_episode_terminal(self, sample):
        for i in range(self.num_envs):
            self.buffers[i].add_sample_until_episode_terminal([s[i] for s in sample])

    def sample(self, batch_size, seq_len=1):
        per_env_batch = np.bincount(
            np.random.randint(0, self.num_envs, batch_size), minlength=self.num_envs
        )
        per_env_samples = []
        for i, batch_sizes in enumerate(per_env_batch):
            if batch_sizes > 0:
                per_env_samples.append(self.buffers[i].sample(batch_sizes, seq_len))
        samples = []
        for i in range(self.num_items):
            samples.append(np.concatenate([sample[i] for sample in per_env_samples], axis=0))

        return samples

    def sample_as_tensors(self, device, batch_size, seq_len=1):
        samples = self.sample(batch_size, seq_len)

        samples_tensor = [
            torch.from_numpy(sample).pin_memory().to(device, non_blocking=True).float()
            for sample in samples
        ]
        return samples_tensor

    def save(self, filepath):
        global_state = {}

        for i, buf in enumerate(self.buffers):
            env_state = buf.get_state()
            for key, value in env_state.items():
                global_state[f"env_{i}_{key}"] = value

        np.savez_compressed(filepath, **global_state)

    def load(self, filepath):
        with np.load(filepath, allow_pickle=True) as global_state:
            for i, buf in enumerate(self.buffers):
                prefix = f"env_{i}_"
                env_state = {
                    key.replace(prefix, ""): global_state[key]
                    for key in global_state.files
                    if key.startswith(prefix)
                }

                buf.set_state(env_state)


class PerEnvTrajectoryBuffer(PerEnvBuffer):
    def __init__(
        self,
        num_envs,
        buffer_shapes=None,
        dtypes=None,
        buffer_size=1_000_000,
        prioritized=False,
        **kwargs,
    ):
        super().__init__(
            num_envs=num_envs,
            buffer_class=TrajectoryReplayBuffer,
            buffer_shapes=buffer_shapes,
            dtypes=dtypes,
            buffer_size=buffer_size,
            prioritized=prioritized,
            **kwargs,
        )

    def sample_with_goals(self, batch_size, seq_len=1, discount=0.99):
        per_env_batch = np.bincount(
            np.random.randint(0, self.num_envs, batch_size), minlength=self.num_envs
        )

        per_env_samples, per_env_goals, per_env_traj_ids = [], [], []
        for i, env_batch_size in enumerate(per_env_batch):
            if env_batch_size > 0:
                data, goal, traj_ids = self.buffers[i].sample_with_goals(
                    env_batch_size, seq_len, discount
                )
                per_env_samples.append(data)
                per_env_goals.append(goal)

                global_ids = np.stack([np.full(env_batch_size, i), traj_ids], axis=1)
                per_env_traj_ids.append(global_ids)

        # determine how many items the sub-buffer actually returned
        num_returned_items = len(per_env_samples[0])

        samples = [
            np.concatenate([s[i] for s in per_env_samples], axis=0)
            for i in range(num_returned_items)
        ]
        goals = np.concatenate(per_env_goals, axis=0)
        traj_ids = np.concatenate(per_env_traj_ids, axis=0)
        return samples, goals, traj_ids

    def sample_with_goals_as_tensors(self, device, batch_size, seq_len=1, discount=0.99):
        samples, goals, traj_ids = self.sample_with_goals(batch_size, seq_len, discount)
        samples_tensors = [
            torch.from_numpy(sample).pin_memory().to(device, non_blocking=True).float()
            for sample in samples
        ]
        goals_tensor = torch.tensor(goals).to(device).float()
        traj_ids = torch.tensor(traj_ids).to(device)
        return samples_tensors, goals_tensor, traj_ids
