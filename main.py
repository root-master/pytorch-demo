from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torchvision import datasets
import time
class Net(nn.Module):
    '''neural network (model)'''
    def __init__(self):
        super(Net, self).__init__()
        # first convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # second convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # dropout random weights
        self.conv2_drop = nn.Dropout2d()
        # fully connected # 1 (layer #3)
        self.fc1 = nn.Linear(320, 50)
        # fully connected # 2 (layer #4)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        '''neural network forward path'''
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    # model is in a train mode -- dropout is applied
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # send data to device (cuda:0 if we use gpu, otherwise cpu)
        data, target = data.to(device), target.to(device)
        # before backward path set gradients to zero 
        optimizer.zero_grad()
        # output of the model = prediction
        output = model(data)
        # loss function = the distance between model's prediction and true labels 
        loss = F.nll_loss(output, target)
        # backward path -- compute gradients 
        loss.backward()
        # apply the optimization step -- change model's parameters
        optimizer.step()
        # print loss and accuracy
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    # model is in a test mode -- dropout is not applied
    model.eval()
    test_loss = 0
    correct = 0
    # no gradients calculations in test mode
    with torch.no_grad():
        for data, target in test_loader:
            # send data (images and labels) to the device
            data, target = data.to(device), target.to(device)
            # output of model = prediction
            output = model(data)
            # calculate test loss
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # convert to one-hot vector
            # a vector with a same dimension of output (number of classes or labels)
            # all zero, but the corresponding predicted label is one 
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # average of all losses = loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # use gpu if available otherwise use cpu
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # load training data (images and labels)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=T.Compose([
                           T.ToTensor(),
                           T.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    # load testing data (images and labels)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=T.Compose([
                           T.ToTensor(),
                           T.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # create a model and send it to the device
    model = Net().to(device)
    # pass model's parameters to optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # run epochs of training and test at the end of each epoch
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time-start_time)