from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import random
from pathlib import Path
from typing import Callable


class AdvantageLerobotDataset(LeRobotDataset):
    """LeRobot dataset for advantage estimator training: history frames, timestep-difference mode, stage progress."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = 'pyav',
    ):
        """Init same as LeRobotDataset; adds episode_index -> array index mapping."""
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend
        )
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes)} if episodes else {}

    def get_sample_with_imgs_from_idx(self, idx: int) -> dict:
        """Return one sample with decoded video frames at index idx."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
            query_indices, padding = self._get_query_indices(idx, arr_idx)
        
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            
            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
            except Exception as e:
                print(f"Error decoding video frames for ep_idx {ep_idx} at idx {idx}: {e}")
                import pdb; pdb.set_trace()
            
            item = {**video_frames, **item}
        
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        
        return item

    def __getitem__(self, idx: int) -> dict:
        """Return sample with delta indices, random timestep comparison frame, and progress label."""
        episode_level_dict = {}
        item_sequence = []
        final_item = {}
        
        # Get main sample
        item = self.get_sample_with_imgs_from_idx(idx)
        
        ep_idx = item["episode_index"].item()
        cur_timestamp = item["timestamp"].item()
        
        _EP_IDX = ep_idx
        _CUR_TIMESTAMP = cur_timestamp
        
        episode_level_dict['episode_length'] = self.meta.episodes[ep_idx]["length"]
        
        # Handle delta indices for action sequences
        if self.delta_indices is not None:
            episode_level_dict = self.handle_delta_indices(idx, ep_idx, episode_level_dict)
        
        # Add task as a string
        task_idx = item["task_index"].item()
        episode_level_dict["task"] = self.meta.tasks[task_idx]
        
        item_sequence.append(item.copy())
        final_item = self.handle_timestep_difference_mode(idx, ep_idx, final_item, _EP_IDX, _CUR_TIMESTAMP)
        final_item = {**final_item, **item_sequence[-1], **episode_level_dict}
        stage_progress_gt_random = final_item[f"his_-100_stage_progress_gt"].item()
        stage_progress_gt = final_item[f"stage_progress_gt"].item()
        final_item['progress'] = stage_progress_gt - stage_progress_gt_random

        return final_item

    def handle_delta_indices(self, idx, ep_idx, episode_level_dict) -> dict:
        """Fill episode_level_dict with action sequence from delta indices."""
        query_indices = None
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        query_indices, padding = self._get_query_indices(idx, arr_idx)
        query_result = self._query_hf_dataset(query_indices)
        
        episode_level_dict = {**episode_level_dict, **padding}
        for key, val in query_result.items():
            episode_level_dict[key] = val
        return episode_level_dict

    def handle_timestep_difference_mode(self, idx, ep_idx, final_item, _EP_IDX, _CUR_TIMESTAMP) -> dict:
        """Add a random same-episode timestep sample with keys prefixed by his_-100_."""
        random_timestep_name = -100
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        ep_start_idx = self.episode_data_index["from"][arr_idx].item()
        ep_end_idx = self.episode_data_index["to"][arr_idx].item()
        while True:
            random_idx = random.randint(ep_start_idx, ep_end_idx - 1)
            if random_idx == idx:
                continue
            
            random_item = self.get_sample_with_imgs_from_idx(random_idx)
            
            ep_idx_check = random_item["episode_index"].item()
            cur_timestamp_check = random_item["timestamp"].item()
            if ep_idx_check != _EP_IDX or cur_timestamp_check == _CUR_TIMESTAMP:
                print(f"Randomly selected invalid timestep, re-sampling. For global idx: {random_idx}, ep_idx: {ep_idx_check}, cur_timestamp: {cur_timestamp_check}")
                continue
            break
        
        _keys = list(random_item.keys())
        for key in _keys:
            new_key = f"his_{random_timestep_name}_{key}"
            random_item[new_key] = random_item.pop(key)

        final_item = {**final_item, **random_item}
        return final_item
