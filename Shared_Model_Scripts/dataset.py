from Datasets import (jesterDS, yt_asl) 

def get_train_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.database in ['jester', 'yt_asl']
    subset = 'training'
    
    if opt.database == 'jester':
        training_set = jesterDS(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.database == 'yt_asl':
        training_set = yt_asl(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    
    return training_set

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    
    return validation_set
    
def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    
    return test_set