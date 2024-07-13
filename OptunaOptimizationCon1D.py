# needed imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from mnist1d.data import make_dataset, get_dataset_args
from Convo1D import Convo
import optuna


def objective(trial):
    batch_size = trial.suggest_int('batch_size', 20, 50)
    learning_rate = trial.suggest_float('learning_rate', .01, .2)
    momentum = trial.suggest_float('momentum', .5, 1)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    step_size = trial.suggest_int('step_size', 10, 40)
    gamma = trial.suggest_float('gamma', .8, .99)
    n_epoch = trial.suggest_int('n_epochs', 100, 200)
    kernel_size = 3
    channels = trial.suggest_int('channels', 10, 20)
    stride = 2

    args = get_dataset_args()
    data = make_dataset(args)
    train_data_x = data['x'].transpose()
    train_data_y = data['y']
    val_data_x = data['x_test'].transpose()
    val_data_y = data['y_test']
    x_train = torch.tensor(train_data_x.transpose().astype('float32'))
    y_train = torch.tensor(train_data_y.astype('long')).long()
    x_val = torch.tensor(val_data_x.transpose().astype('float32'))
    y_val = torch.tensor(val_data_y.astype('long')).long()
    data_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    # move model to gpu
    model = Convo(stride=2, channels=channels, kernel_size=3).to('cuda')
    # initialize weights
    model.weights_init()
    # set optimizer to SGD with adjustable hyperparams
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # set loss to cross entropy
    lossfn = nn.CrossEntropyLoss()
    # set step lr adjuster with respective hyperparams to be adjusted
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.train()
    for epoch in range(n_epoch):
        # loop over batches
        for i, data in enumerate(data_loader):
            # retrieve inputs and labels for this batch and move to gpu
            x_batch, y_batch = data
            x_batch = x_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            pred = model(x_batch[:, None, :])
            # compute the loss
            loss = lossfn(pred, y_batch)
            # backward pass
            loss.backward()
            # SGD update
            optimizer.step()

        scheduler.step()
    model.eval()
    with torch.no_grad():
        x_val = x_val.to('cuda')
        y_val = y_val.to('cuda')
        valPred = model(x_val[:, None, :])
        lossval = lossfn(valPred, y_val)
    return lossval.item()


def optimizeit(trials):
    study = optuna.create_study(direction='minimize', study_name='MNIST')
    study.optimize(objective, n_trials=trials)

