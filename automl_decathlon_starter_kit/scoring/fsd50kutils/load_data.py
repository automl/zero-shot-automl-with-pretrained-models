
from .audio_dataset import *


'''
Returns a dataset for FSD50K
'''
def load_fsd50k_dataset(root, split):
    assert split=="train" or split=="val" or split=="test", \
        "split should be one of 'train' or 'val' or 'test'"

    audio_config = {
        'feature': 'melspectrogram',
        'sample_rate': 22050,
        'min_duration': 1,
    }


    if split=="train":
        mixer = BackgroundAddMixer()
        train_mixer = UseMixerWithProb(mixer, 0.75)
        train_transforms = get_transforms_fsd_chunks(True, 101)
        train_set = \
            SpectrogramDataset( root=root,
                                audio_config=audio_config,
                                mixer=train_mixer,
                                transform=train_transforms,
                              )
        return train_set
    else:
        valtest_transforms = get_transforms_fsd_chunks(False, 101)
        valtest_set = \
            FSD50kEvalDataset( root=root,
                               split=split,
                               audio_config=audio_config,
                               transform=valtest_transforms
                               
                             )
        return valtest_set
