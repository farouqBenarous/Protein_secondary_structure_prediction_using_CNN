import numpy as np
import gzip


def load_dataset(dataset_path, dataset_meta_data):
    ds = np.load(dataset_path)
    ds = np.reshape(ds, (ds.shape[0], dataset_meta_data['sequence_len'], dataset_meta_data['total_features']))
    ret = np.zeros(
        (ds.shape[0], ds.shape[1], dataset_meta_data['amino_acid_residues'] + dataset_meta_data['num_classes']))
    ret[:, :, 0:dataset_meta_data['amino_acid_residues']] = ds[:, :, 35:56]
    ret[:, :, dataset_meta_data['amino_acid_residues']:] = ds[:, :, dataset_meta_data['amino_acid_residues'] + 1:
                                                                    dataset_meta_data['amino_acid_residues'] + 1 +
                                                                    dataset_meta_data['num_classes']]
    return ret


def read_and_process(dataset,dataset_meta_data):
    X = dataset[:, :, 0:dataset_meta_data['amino_acid_residues']]
    Y = dataset[:, :, dataset_meta_data['amino_acid_residues']:dataset_meta_data['amino_acid_residues'] + dataset_meta_data['num_classes']]
    return X, Y
