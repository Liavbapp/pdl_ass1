
class HyperParams:
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 100

    # batches_sizes = [2 ** i for i in range(0, 7)]
    # lrs = [1 * (1/10)**i for i in range(2, 6)]
class DataSets:
    swiss_roll = 'SwissRollData'
    gmm = 'GMMData'
    peaks = 'PeaksData.mat'
