import typing
import numpy as np
obs_names = ['phase', 'player_hp', 'player_max_hp', 'player_sp', 'player_max_sp', 'boss_hp', 'boss_max_hp', 'player_pose', 'boss_pose',
    'camera_pose',  'player_animation', 'player_animation_duration', 'boss_animation', 'boss_animation_duration', 'lock_on']
obs_lengths = [1, 1, 1, 1, 1,1,1,4,4,6, 1, 1, 1, 1, 1]
def obs2numpy(data:typing.Mapping):
    assert len(obs_names) == len(obs_lengths)
    flattened_data = np.concatenate((
        [data['phase']],
        data['player_hp'],
        [data['player_max_hp']],
        data['player_sp'],
        [data['player_max_sp']],
        data['boss_hp'],
        [data['boss_max_hp']],
        data['player_pose'],
        data['boss_pose'],
        data['camera_pose'],
        [data['player_animation']],
        data['player_animation_duration'],
        [data['boss_animation']],
        data['boss_animation_duration'],
        [float(data['lock_on'])]
    )).astype(np.float32).reshape(-1)
    return flattened_data