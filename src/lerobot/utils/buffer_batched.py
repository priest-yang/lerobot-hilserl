#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import functools
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or scalar value.
    If it looks like a 3D (C,H,W) or (H,W,C) shape, we might consider it an 'image'.
    Otherwise default to appropriate dtype for numeric.
    """

    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and (shape[0] in [1, 3] or shape[2] in [1, 3]):
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """
    Concatenates two BatchTransition objects into one.

    This function merges the right BatchTransition into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransition): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransition): The second batch to append to the first one.

    Returns:
        BatchTransition: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # Concatenate basic fields
    left_batch_transitions["action"] = torch.cat(
        [left_batch_transitions["action"], right_batch_transition["action"]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # Only process if right_info exists
    if right_info is not None:
        # Initialize left complementary_info if needed
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # Concatenate each field
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions


class ParallelReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ):
        """
        Parallel replay buffer for storing transitions from multiple environments.
        Storage shape: (num_envs, capacity, ...) instead of (capacity, ...)
        
        Args:
            capacity (int): Maximum number of transitions to store per environment.
            num_envs (int): Number of parallel environments.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")
        if num_envs <= 0:
            raise ValueError("Number of environments must be greater than 0.")

        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.storage_device = storage_device
        # Position tracking for each environment: [num_envs]
        self.position = torch.zeros(num_envs, dtype=torch.long, device=storage_device)
        # Size tracking for each environment: [num_envs]
        self.size = torch.zeros(num_envs, dtype=torch.long, device=storage_device)
        self.initialized = False
        self.optimize_memory = optimize_memory

        # Track episode boundaries for memory optimization: (num_envs, capacity)
        self.episode_ends = torch.zeros((num_envs, capacity), dtype=torch.bool, device=storage_device)

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        self.image_augmentation_function = image_augmentation_function

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        # For parallel buffer, we need to get the shape per environment (remove batch dimension)
        state_shapes = {key: val[0].shape for key, val in state.items()}
        action_shape = action[0].shape

        # Pre-allocate tensors for storage with parallel dimension: (num_envs, capacity, ...)
        self.states = {
            key: torch.empty((self.num_envs, self.capacity, *value[0].shape), device=self.storage_device, dtype=value.dtype)
            for key, value in state.items()
        }
        self.actions = torch.empty((self.num_envs, self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.empty((self.num_envs, self.capacity), device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
            key: torch.empty((self.num_envs, self.capacity, *value[0].shape), device=self.storage_device, dtype=value.dtype)
            for key, value in state.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.empty((self.num_envs, self.capacity), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.num_envs, self.capacity), dtype=torch.bool, device=self.storage_device)

        # Initialize storage for complementary_info
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # Pre-allocate tensors for each key in complementary_info
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value[0].shape
                    self.complementary_info[key] = torch.empty(
                        (self.num_envs, self.capacity, *value_shape), device=self.storage_device, dtype=value.dtype
                    )
                elif isinstance(value, (int, float)):
                    # Handle scalar values similar to reward
                    self.complementary_info[key] = torch.empty((self.num_envs, self.capacity), device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def __len__(self):
        return self.size.sum().item()

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: dict[str, torch.Tensor],
        done: torch.Tensor,
        truncated: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """
        Saves transitions for all parallel environments.
        
        Args:
            state: dict of tensors with shape (num_envs, ...)
            action: tensor with shape (num_envs, ...)
            reward: tensor with shape (num_envs,)
            next_state: dict of tensors with shape (num_envs, ...)
            done: tensor with shape (num_envs,)
            truncated: tensor with shape (num_envs,)
            complementary_info: dict of tensors with shape (num_envs, ...) or None
        """
        # Initialize storage if this is the first transition
        if not self.initialized:
            self._initialize_storage(state=state, action=action, complementary_info=complementary_info)

        # Store the transitions in pre-allocated tensors
        for key in self.states:
            # Store states: (num_envs, capacity, ...)
            self.states[key][torch.arange(self.num_envs), self.position] = state[key]

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_states[key][torch.arange(self.num_envs), self.position] = next_state[key]

        # Store other tensors
        self.actions[torch.arange(self.num_envs), self.position] = action
        self.rewards[torch.arange(self.num_envs), self.position] = reward
        self.dones[torch.arange(self.num_envs), self.position] = done
        self.truncateds[torch.arange(self.num_envs), self.position] = truncated

        # Handle complementary_info if provided and storage is initialized
        if complementary_info is not None and self.has_complementary_info:
            # Store the complementary_info
            for key in self.complementary_info_keys:
                if key in complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][torch.arange(self.num_envs), self.position] = value
                    elif isinstance(value, (int, float)):
                        self.complementary_info[key][torch.arange(self.num_envs), self.position] = value

        # Update positions and sizes for each environment
        self.position = (self.position + 1) % self.capacity
        self.size = torch.min(self.size + 1, torch.tensor(self.capacity, device=self.storage_device))

    def sample(self, batch_size: int) -> BatchTransition:
        """
        Sample a random batch of transitions from all environments uniformly.
        """
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        # Calculate total available transitions across all environments
        total_size = self.size.sum().item()
        if total_size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        batch_size = min(batch_size, total_size)

        # Sample uniformly from all environments
        # First, determine how many samples to take from each environment
        samples_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.storage_device)
        
        # Distribute batch_size samples across environments proportionally
        remaining_batch = batch_size
        for env_idx in range(self.num_envs):
            if remaining_batch <= 0:
                break
            available_in_env = self.size[env_idx].item()
            if available_in_env > 0:
                # Take proportional samples from this environment
                env_samples = min(available_in_env, remaining_batch)
                samples_per_env[env_idx] = env_samples
                remaining_batch -= env_samples

        # Generate random indices for each environment
        env_indices = []
        for env_idx in range(self.num_envs):
            if samples_per_env[env_idx] > 0:
                available_size = self.size[env_idx].item()
                if available_size > 0:
                    # Generate random indices within available data
                    env_idx_tensor = torch.randint(low=0, high=available_size, size=(samples_per_env[env_idx],), device=self.storage_device)
                    env_indices.append((env_idx, env_idx_tensor))

        # Identify image keys that need augmentation
        image_keys = [k for k in self.states if k.startswith("observation.image")] if self.use_drq else []

        # Create batched state and next_state
        batch_state = {}
        batch_next_state = {}

        # First pass: load all state tensors to target device
        for key in self.states:
            batch_state[key] = []
            batch_next_state[key] = []

            for env_idx, env_idx_tensor in env_indices:
                # Convert random indices to actual buffer positions
                # For circular buffer: actual_idx = (position - size + random_idx) % capacity
                actual_indices = (self.position[env_idx] - self.size[env_idx] + env_idx_tensor) % self.capacity
                
                # Load states from this environment
                env_states = self.states[key][env_idx][actual_indices].to(self.device)
                batch_state[key].append(env_states)

                if not self.optimize_memory:
                    # Standard approach - load next_states directly
                    env_next_states = self.next_states[key][env_idx][actual_indices].to(self.device)
                    batch_next_state[key].append(env_next_states)
                else:
                    # Memory-optimized approach - get next_state from the next index
                    next_actual_indices = (actual_indices + 1) % self.capacity
                    env_next_states = self.states[key][env_idx][next_actual_indices].to(self.device)
                    batch_next_state[key].append(env_next_states)

            # Concatenate across environments
            batch_state[key] = torch.cat(batch_state[key], dim=0)
            batch_next_state[key] = torch.cat(batch_next_state[key], dim=0)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # Optimization: Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # Calculate offsets for the current image key:
                # For each key, we have 2*batch_size images (batch_size for states, batch_size for next_states)
                # States start at index i*2*batch_size and take up batch_size slots
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next states start after the states at index (i*2+1)*batch_size and also take up batch_size slots
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # Sample other tensors
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_truncateds = []

        for env_idx, env_idx_tensor in env_indices:
            # Convert random indices to actual buffer positions
            actual_indices = (self.position[env_idx] - self.size[env_idx] + env_idx_tensor) % self.capacity
            
            batch_actions.append(self.actions[env_idx][actual_indices].to(self.device))
            batch_rewards.append(self.rewards[env_idx][actual_indices].to(self.device))
            batch_dones.append(self.dones[env_idx][actual_indices].to(self.device).float())
            batch_truncateds.append(self.truncateds[env_idx][actual_indices].to(self.device).float())

        batch_actions = torch.cat(batch_actions, dim=0)
        batch_rewards = torch.cat(batch_rewards, dim=0)
        batch_dones = torch.cat(batch_dones, dim=0)
        batch_truncateds = torch.cat(batch_truncateds, dim=0)

        # Sample complementary_info if available
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = []
                for env_idx, env_idx_tensor in env_indices:
                    # Convert random indices to actual buffer positions
                    actual_indices = (self.position[env_idx] - self.size[env_idx] + env_idx_tensor) % self.capacity
                    batch_complementary_info[key].append(self.complementary_info[key][env_idx][actual_indices].to(self.device))
                batch_complementary_info[key] = torch.cat(batch_complementary_info[key], dim=0)

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)

        Yields:
            BatchTransition: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size)

            # Yield all items from the iterator
            with suppress(StopIteration):
                yield from iterator

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Create an iterator that continuously yields prefetched batches in a
        background thread. The design is intentionally simple and avoids busy
        waiting / complex state management.

        Args:
            batch_size (int): Size of batches to sample.
            queue_size (int): Maximum number of prefetched batches to keep in
                memory.

        Yields:
            BatchTransition: A batch sampled from the replay buffer.
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """Continuously put sampled batches into the queue until shutdown."""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size)
                    # The timeout ensures the thread unblocks if the queue is full
                    # and the shutdown event gets set meanwhile.
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # Queue is full – loop again (will re-check shutdown_event)
                    continue
                except Exception:
                    # Surface any unexpected error and terminate the producer.
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # If the producer already set the shutdown flag we exit.
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # Drain the queue quickly to help the thread exit if it's blocked on `put`.
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # Give the producer thread a bit of time to finish.
            producer_thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            queue_size (int): Number of initial batches to prefetch

        Yields:
            BatchTransition: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        num_envs: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ) -> "ParallelReplayBuffer":
        """
        Convert a LeRobotDataset into a ParallelReplayBuffer.
        Episodes are distributed evenly across environments.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            num_envs (int): Number of parallel environments.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity per environment. If None, uses dataset length / num_envs.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.

        Returns:
            ParallelReplayBuffer: The replay buffer with dataset transitions.
        """
        if capacity is None:
            capacity = len(lerobot_dataset) // num_envs
            if capacity == 0:
                capacity = 1

        if capacity < len(lerobot_dataset) // num_envs:
            raise ValueError(
                "The capacity of the ParallelReplayBuffer must be greater than or equal to the length of the LeRobotDataset divided by num_envs."
            )

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            num_envs=num_envs,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        # Convert dataset to transitions
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

        # Group transitions by episodes
        episodes = []
        current_episode = []
        for transition in list_transition:
            current_episode.append(transition)
            if transition["done"]:
                episodes.append(current_episode)
                current_episode = []
        
        # Add the last episode if it's not empty
        if current_episode:
            episodes.append(current_episode)

        # Distribute episodes across environments
        env_episodes = [[] for _ in range(num_envs)]
        for i, episode in enumerate(episodes):
            env_idx = i % num_envs
            env_episodes[env_idx].extend(episode)

        # Initialize the buffer with the first transition to set up storage tensors
        if list_transition:
            first_transition = list_transition[0]
            first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
            first_action = first_transition["action"].to(device)

            # Get complementary info if available
            first_complementary_info = None
            if (
                "complementary_info" in first_transition
                and first_transition["complementary_info"] is not None
            ):
                first_complementary_info = {
                    k: v.to(device) for k, v in first_transition["complementary_info"].items()
                }

            replay_buffer._initialize_storage(
                state=first_state, action=first_action, complementary_info=first_complementary_info
            )

        # Fill the buffer with transitions distributed across environments
        for env_idx in range(num_envs):
            for transition in env_episodes[env_idx]:
                # Add to specific environment
                replay_buffer._add_to_env(
                    env_idx=env_idx,
                    state=transition["state"],
                    action=transition["action"],
                    reward=transition["reward"],
                    next_state=transition["next_state"],
                    done=transition["done"],
                    truncated=transition["truncated"],
                    complementary_info=transition["complementary_info"],
                )

        return replay_buffer

    def _add_to_env(
        self,
        env_idx: int,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: dict[str, torch.Tensor],
        done: torch.Tensor,
        truncated: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Add a single transition to a specific environment."""
        # Store the transition in pre-allocated tensors for the specific environment
        for key in self.states:
            self.states[key][env_idx, self.position[env_idx]].copy_(state[key].squeeze(0))

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_states[key][env_idx, self.position[env_idx]].copy_(next_state[key].squeeze(0))

        self.actions[env_idx, self.position[env_idx]] = action.squeeze(0)
        self.rewards[env_idx, self.position[env_idx]] = reward.squeeze(0)
        self.dones[env_idx, self.position[env_idx]] = done.squeeze(0)
        self.truncateds[env_idx, self.position[env_idx]] = truncated.squeeze(0)

        # Handle complementary_info if provided and storage is initialized
        if complementary_info is not None and self.has_complementary_info:
            # Store the complementary_info
            for key in self.complementary_info_keys:
                if key in complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][env_idx, self.position[env_idx]] = value.squeeze(0)
                    elif isinstance(value, (int, float)):
                        self.complementary_info[key][env_idx, self.position[env_idx]] = value

        # Update position and size for the specific environment
        self.position[env_idx] = (self.position[env_idx] + 1) % self.capacity
        self.size[env_idx] = min(self.size[env_idx] + 1, self.capacity)

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_parallel_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ParallelReplayBuffer into a single LeRobotDataset object.
        """
        total_size = self.size.sum().item()
        if total_size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        sample_action = self.actions[0, 0]  # First environment, first position
        act_info = guess_feature_info(t=sample_action, name="action")
        features["action"] = act_info

        # Add "reward" and "done"
        features["next.reward"] = {"dtype": "float32", "shape": (1,)}
        features["next.done"] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.states:
            sample_val = self.states[key][0, 0]  # First environment, first position
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # Add complementary_info keys if available
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                sample_val = self.complementary_info[key][0, 0]  # First environment, first position
                if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
                    sample_val = sample_val.unsqueeze(0)
                f_info = guess_feature_info(t=sample_val, name=f"complementary_info.{key}")
                features[f"complementary_info.{key}"] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # Start writing images if needed
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # Convert transitions into episodes and frames
        global_frame_idx = 0
        episode_idx = 0

        for env_idx in range(self.num_envs):
            env_size = self.size[env_idx].item()
            if env_size == 0:
                continue

            # Track current episode data
            current_episode_frames = []
            episode_is_success = False

            for frame_idx in range(env_size):
                actual_idx = (self.position[env_idx] - env_size + frame_idx) % self.capacity

                frame_dict = {}

                # Fill the data for state keys
                for key in self.states:
                    frame_dict[key] = self.states[key][env_idx, actual_idx].cpu()

                # Fill action, reward, done
                frame_dict["action"] = self.actions[env_idx, actual_idx].cpu()
                frame_dict["next.reward"] = torch.tensor([self.rewards[env_idx, actual_idx]], dtype=torch.float32).cpu()
                frame_dict["next.done"] = torch.tensor([self.dones[env_idx, actual_idx]], dtype=torch.bool).cpu()
                frame_dict["task"] = task_name

                # Add complementary_info if available
                if self.has_complementary_info:
                    for key in self.complementary_info_keys:
                        val = self.complementary_info[key][env_idx, actual_idx]
                        # Convert tensors to CPU
                        if isinstance(val, torch.Tensor):
                            if val.ndim == 0:
                                val = val.unsqueeze(0)
                            frame_dict[f"complementary_info.{key}"] = val.cpu()
                        # Non-tensor values can be used directly
                        else:
                            frame_dict[f"complementary_info.{key}"] = val

                # Check if this frame indicates success
                if 'complementary_info.is_success' in frame_dict:
                    success_val = frame_dict['complementary_info.is_success']
                    if isinstance(success_val, torch.Tensor) and success_val.item() == 1.0:
                        episode_is_success = True

                # Add frame to current episode
                current_episode_frames.append(frame_dict)

                # If we reached an episode boundary, check if it was successful
                if self.dones[env_idx, actual_idx] or self.truncateds[env_idx, actual_idx]:
                    if episode_is_success:
                        # Only save successful episodes
                        for frame in current_episode_frames:
                            lerobot_dataset.add_frame(frame)
                            global_frame_idx += 1
                        lerobot_dataset.save_episode()
                        episode_idx += 1
                        print(f"Saved successful episode {episode_idx} with {len(current_episode_frames)} frames")
                    
                    # Reset for next episode
                    current_episode_frames = []
                    episode_is_success = False

        #! note: remaining frames will be discarded

        lerobot_dataset.stop_image_writer()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
    ) -> list[BatchTransition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.
        Same implementation as the original ReplayBuffer.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        transitions = []
        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = "next.done" in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample["action"].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = current_sample["next.reward"]

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = current_sample["next.done"]
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # TODO: (azouitine) Check if it's necessary to convert to tensor
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct the BatchTransition -----
            transition = BatchTransition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            transitions.append(transition)

        return transitions
