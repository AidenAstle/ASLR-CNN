from Datasets.wlasl import Wlasl

def get_train_set(spatial_transform, temporal_transform, target_transform):
    training_set = Wlasl(
            "./wlasl_data/extracted", # paths are from the root folder since thats where testing is run from
            "./Build_code/Datasets/Data/wlasl_data.json",
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=16) #default
    
    return training_set

def get_validation_set(spatial_transform, temporal_transform, target_transform):
    validation_set = Wlasl(
            "./wlasl_data/extracted", # paths are from the root folder since thats where testing is run from
            "./Build_code/Datasets/Data/wlasl_data.json",
            'validation',
            1, #num samples, defaults to 1
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=16) #default
    return validation_set
    
def get_test_set(tst_subset, spatial_transform, temporal_transform, target_transform):
    assert tst_subset in ['val', 'test']

    if tst_subset == 'val':
        subset = 'validation'
    elif tst_subset == 'test':
        subset = 'testing'
    test_set = Wlasl(
            "./wlasl_data/extracted", # paths are from the root folder since thats where testing is run from
            "./Build_code/Datasets/Data/wlasl_data.json",
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=16)
    return test_set