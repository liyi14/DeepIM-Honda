def merge_datasets(datasets):
    dataset = datasets[0]
    if len(datasets)>1:
        for d in datasets[1:]:
            dataset._name += '_'+d._image_set
            dataset._image_set += '_'+d._image_set
            dataset._roidb += d._roidb
    dataset.perm_if_needed()
    return dataset