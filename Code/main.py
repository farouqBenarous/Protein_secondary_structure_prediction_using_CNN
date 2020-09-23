import numpy as np
from sklearn.model_selection import train_test_split
from Code.model.cnn import CnnModel
from Code.shared.dataset_functions import load_dataset, read_and_process

if __name__ == '__main__':
    np.random.seed(42)

    sequence_len = 700
    total_features = 57
    amino_acid_residues = 21
    num_classes = 8
    # ---------------- Data loading , processing ---------------#

    print("loading the data ...")
    dataset = load_dataset("../dataset/cullpdb+profile_6133.npy.gz",{'sequence_len': sequence_len, 'amino_acid_residues': amino_acid_residues,
                                 'num_classes': num_classes,'total_features':total_features})

    print("preprocessing the data ...")
    X, Y = read_and_process(dataset,{'sequence_len': sequence_len, 'amino_acid_residues': amino_acid_residues,
                                 'num_classes': num_classes})
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=2)

    # ---------------- Training / testing / infer the models ---------------#

    # Training
    print("CNN  Model is being called for training  ")
    CnnModel.train_CNN_model_V1(X_train, y_train,
                                {'sequence_len': sequence_len, 'amino_acid_residues': amino_acid_residues,
                                 'num_classes': num_classes})

    # Testing
    print("CNN  Model is being called for testing ")
    CnnModel.test(X_test, y_test, "cnnV1.h5")
