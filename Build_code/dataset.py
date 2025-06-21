from Datasets import (jesterDS, wlasl) 

def get_train_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.database in ['jester', 'wlasl']
    subset = 'training'
    
    training_set = wlasl(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    
    return training_set

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    validation_set = wlasl(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_set
    
def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    test_set = wlasl(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return test_set