"""
Pytorch implementation of CapsNet in paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       Launch `python CapsNet.py -h` for usage help

Result:
    Validation accuracy > 99.6% after 50 epochs.
    Speed: About 73s/epoch on a single GTX1070 GPU card and 43s/epoch on a GTX1080Ti GPU.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""
import sys

sys.path.insert(1, './advertorch/advertorch/attacks/')
from one_step_gradient import GradientSignAttack
from iterative_projected_gradient import PGDAttack, LinfPGDAttack
from carlini_wagner import CarliniWagnerL2Attack
from iterative_projected_gradient import LinfBasicIterativeAttack
import torch
import numpy
import numpy as np
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.utils.data as DataUtils

class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 12, kernel_size=6, stride=2, padding=2)

        self.conv2 = nn.Conv2d(12,16,kernel_size=6,stride=2,padding=2)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        
        self.primarycaps = PrimaryCapsule(16,16)
        
        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=49, in_dim_caps=16,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon

def compute_distance(x, x_recon, distance):
  for a,b in zip(x,x_recon):
    distance = torch.cat((distance,torch.norm(a - b).expand(1)))
  return distance  


def show_reconstruction(model, test_loader, n_images, args):
    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image
    import numpy as np
    distance = torch.zeros((1,)).cuda()
    adversary = GradientSignAttack(model, loss_fn=caps_loss, eps=0.3)
    model.eval()
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda()), Variable(y.cuda())
        x_adv = adversary.perturb(x,args.lam_recon,y) 
        _, x_recon = model(x_adv)
        distance = compute_distance(x_adv, x_recon,distance)
        data = np.concatenate([x.cpu().data, x_recon.cpu().data])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
        print('-' * 70)
        plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", ))
        plt.show()
        break
    distance = distance[1:]
    print(distance)    


def threshold(model,test_loader,args):
    model.eval()
    distance = torch.zeros((1,)).cuda()
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)        
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        y_pred, x_recon = model(x)
        distance = compute_distance(x, x_recon,distance)
    distance = distance[1:]
    threshold_val = numpy.percentile(distance.cpu().detach().numpy(),95)
    ind = torch.argmax(distance)
    print(threshold_val)
    return threshold_val

def test(model, test_loader, val_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    #threshold_val = threshold(model,val_loader,args)
    distance = torch.zeros((1,)).cuda()
    #weight = torch.zeros((1,)).cuda()
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda()), Variable(y.cuda())
        y_pred, x_recon = model(x)       
        test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).item() * x.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()
        #weight = torch.cat((weight,~y_pred.eq(y_true)))
        #distance = compute_distance(x, x_recon,distance)
    #distance = distance[1:]
    #weight = weight[1:]
    #undetected_r =  ((distance*weight < threshold_val).cpu()*(distance*weight > 0).cpu()).sum()
    #print("Undetected Rate : {}".format(undetected_r/100))
    test_loss /= 10000
    return test_loss, correct /100

def test_fgsm(model, test_loader, val_loader,args):
    
    test_loss = 0
    incorrect = 0
    adversary = GradientSignAttack(model, loss_fn=caps_loss, eps=0.3)
    model.eval()
    threshold_val = threshold(model,test_loader,args)
    distance = torch.zeros((1,)).cuda()
    weight = torch.zeros((1,)).cuda()
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
        
        x_adv = adversary.perturb(x,args.lam_recon,y)   
        y_pred_adv,x_recon_adv = model(x_adv)    
        test_loss += caps_loss(y, y_pred_adv, x_adv, x_recon_adv, args.lam_recon).item() * x_adv.size(0)  # sum up batch loss
        y_pred_adv = y_pred_adv.data.max(1)[1]
        y_true = y.data.max(1)[1]
        incorrect += (~y_pred_adv.eq(y_true)).cpu().sum()
        weight = torch.cat((weight,y_pred_adv.eq(y_true)))
        distance = compute_distance(x_adv, x_recon_adv,distance)
    distance = distance[1:]
    weight = 1- weight[1:]
    undetected_r = ((distance*weight < threshold_val).cpu()*(distance*weight > 0).cpu()).sum()
    print("Undetected Rate : {}".format(undetected_r/100))
    test_loss /= 10000
    return test_loss, incorrect /100

def test_pgd(model, test_loader, val_loader, args):
    
    test_loss = 0
    incorrect = 0
    adversary = PGDAttack(model, loss_fn=caps_loss, eps=0.3)
    model.eval()
    threshold_val = threshold(model,test_loader,args)
    distance = torch.zeros((1,)).cuda()
    weight = torch.zeros((1,)).cuda()
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda()), Variable(y.cuda())      
        x_adv = adversary.perturb(x,args.lam_recon,y)   
        y_pred_adv,x_recon_adv = model(x_adv)    
        test_loss += caps_loss(y, y_pred_adv, x_adv, x_recon_adv, args.lam_recon).item() * x_adv.size(0)  # sum up batch loss
        y_pred_adv = y_pred_adv.data.max(1)[1]
        y_true = y.data.max(1)[1]
        incorrect += (~y_pred_adv.eq(y_true).cpu()).sum()
        weight = torch.cat((weight,y_pred_adv.eq(y_true)))
        distance = compute_distance(x_adv, x_recon_adv,distance)
    distance = distance[1:]
    print(distance[:10])
    weight = 1 - weight[1:]
    undetected_r = ((distance*weight < threshold_val).cpu()*(distance*weight > 0).cpu()).sum()
    print("Undetected Rate : {}".format(undetected_r/100))
    test_loss /= 10000
    return test_loss, incorrect /100

def test_bim(model, test_loader, val_loader, args):
    
    test_loss = 0
    incorrect = 0
    adversary = LinfBasicIterativeAttack(model, loss_fn=caps_loss, eps=0.3,nb_iter=1000)
    model.eval()
    threshold_val = threshold(model,test_loader,args)
    distance = torch.zeros((1,)).cuda()
    weight = torch.zeros((1,)).cuda()
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.) 
        x, y = Variable(x.cuda()), Variable(y.cuda()) 
        x_adv = adversary.perturb(x,args.lam_recon,y)   
        y_pred_adv,x_recon_adv = model(x_adv)    
        test_loss += caps_loss(y, y_pred_adv, x_adv, x_recon_adv, args.lam_recon).item() * x_adv.size(0)  # sum up batch loss
        y_pred_adv = y_pred_adv.data.max(1)[1]
        y_true = y.data.max(1)[1]
        incorrect += (~y_pred_adv.eq(y_true).cpu()).sum()
        weight = torch.cat((weight,y_pred_adv.eq(y_true)))
        distance = compute_distance(x_adv, x_recon_adv,distance)
    distance = distance[1:]
    weight = 1 - weight[1:]
    undetected_r = undetected_r = ((distance*weight < threshold_val).cpu()*(distance*weight > 0).cpu()).sum()
    print("Undetected Rate : {}".format(undetected_r/100))
    test_loss /= 10000
    return test_loss, incorrect /100

def test_cw(model, test_loader,val_loader, args):
    
    test_loss = 0
    incorrect = 0
    adversary = CarliniWagnerL2Attack(model, num_classes = 10,max_iterations=1000)
    model.eval()
    threshold_val = threshold(model,val_loader,args)
    distance = torch.zeros((1,)).cuda()
    weight = torch.zeros((1,)).cuda()
    for i,(x, y) in enumerate(test_loader):
        print("Progress {}".format(i*100/len(test_loader)))
        y_i = y.detach().clone()
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
        
        x_adv = adversary.perturb(x,y_i.cuda())   
        y_pred_adv,x_recon_adv = model(x_adv,y)    
        test_loss += caps_loss(y, y_pred_adv, x_adv, x_recon_adv, args.lam_recon).item() * x_adv.size(0)  # sum up batch loss
        y_pred_adv = y_pred_adv.data.max(1)[1]
        y_true = y.data.max(1)[1]
        incorrect += (~y_pred_adv.eq(y_true).cpu()).sum()
        weight = torch.cat((weight,~y_pred_adv.eq(y_true)))
        distance = compute_distance(x_adv, x_recon_adv,distance)
    distance = distance[1:]
    weight = weight[1:]
    undetected_r = ((distance*weight < threshold_val).cpu()*(distance*weight > 0).cpu()).sum()
    print("Undetected Rate : {}".format(undetected_r/100))
    test_loss /= 10000
    return test_loss, incorrect /100

def train(model, train_loader, val_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-'*70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    #lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        #lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward

            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        # compute validation loss and acc
        val_loss, val_acc = test(model, val_loader,val_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model


def load_dataset(path='./data', download=False, batch_size=100, shift_pixels=2,dataset=None):
    """
    Construct dataloaders for training and test data. Data augmentation is also done here.
    :param path: file path of the dataset
    :param download: whether to download the original data
    :param batch_size: batch size
    :param shift_pixels: maximum number of pixels to shift in each direction
    :return: train_loader, test_loader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if dataset == 'SVHN': 
        trainSet = datasets.SVHN(root=path, download=download, train=True, \
                           transform=transforms.Compose([transforms.RandomCrop(size=32, padding=shift_pixels),
                                                        transforms.ToTensor()]))
        valSet = datasets.SVHN(root=path, download=download, train=True, \
                         transform=transforms.Compose([transforms.RandomCrop(size=32, padding=shift_pixels),
                                                        transforms.ToTensor()]))
        testSet = datasets.SVHN(root=path, download=download, train=False, \
                                 transform=transforms.Compose([transforms.RandomCrop(size=32, padding=shift_pixels),
                                                        transforms.ToTensor()]))
  
        indices = np.arange(0, 73257)
        #np.random.shuffle(indices)
  
        trainSampler = SubsetRandomSampler(indices[:61257])
        valSampler = SequentialSampler(indices[61257:])
        testSampler = SequentialSampler(np.arange(0, 26032))
        
        train_loader = DataUtils.DataLoader(trainSet, batch_size=batch_size, \
                                        sampler=trainSampler)
        val_loader = DataUtils.DataLoader(valSet, batch_size=batch_size, \
                                        sampler=valSampler)
        test_loader = DataUtils.DataLoader(testSet, batch_size=batch_size, \
                                          sampler=testSampler) 
    elif dataset == 'FashionMNIST' :
        trainSet = datasets.FashionMNIST(root=path, download=download, train=True, \
                           transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                        transforms.ToTensor()]))
        valSet = datasets.FashionMNIST(root=path, download=download, train=True, \
                         transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                        transforms.ToTensor()]))
        testSet = datasets.FashionMNIST(root=path, download=download, train=False, \
                                 transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                        transforms.ToTensor()]))
  
        indices = np.arange(0, 60000)
        #np.random.shuffle(indices)
  
        trainSampler = SubsetRandomSampler(indices[:50000])
        valSampler = SequentialSampler(indices[50000:])
        
        
        train_loader = DataUtils.DataLoader(trainSet, batch_size=batch_size, \
                                        sampler=trainSampler)
        val_loader = DataUtils.DataLoader(valSet, batch_size=batch_size, \
                                        sampler=valSampler)
        test_loader = DataUtils.DataLoader(testSet, batch_size=batch_size, \
                                          shuffle=False) 
    else:
        trainSet = datasets.MNIST(root=path, download=download, train=True, \
                           transform=transforms.Compose([
                                                        transforms.ToTensor()]))
        valSet = datasets.MNIST(root=path, download=download, train=True, \
                         transform=transforms.Compose([
                                                        transforms.ToTensor()]))
        testSet = datasets.MNIST(root=path, download=download, train=False, \
                                 transform=transforms.Compose([
                                                        transforms.ToTensor()]))
  
        indices = np.arange(0, 60000)
        #np.random.shuffle(indices)
  
        trainSampler = SubsetRandomSampler(indices[:60000])
        #valSampler = SequentialSampler(indices[50000:])
        #testSampler = SequentialSampler(np.arange(0, 10000))
        
        train_loader = DataUtils.DataLoader(trainSet, batch_size=batch_size, \
                                        sampler=trainSampler )
        val_loader = DataUtils.DataLoader(valSet, batch_size=batch_size, \
                                        )
        test_loader = DataUtils.DataLoader(testSet, batch_size=batch_size, \
                                          )   
       
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-fg', '--fgsm', action='store_true',
                        help="Test the trained model on FGSM adversarial testing dataset")
    parser.add_argument('-pg', '--pgd', action='store_true',
                        help="Test the trained model on PGD adversarial testing dataset")  
    parser.add_argument('-bi', '--bim', action='store_true',
                        help="Test the trained model on BIM adversarial testing dataset")                        
    parser.add_argument('-cw', '--cwl2', action='store_true',
                        help="Test the trained model on CW adversarial testing dataset")                                                      
    parser.add_argument('-th', '--threshold', action='store_true',
                        help="Compute the threshold `theta` ")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--dataset',default='mnist',help='Dataset to be used for training and testing. Dataset name should be provided as MNIST, FashionMNIST or SVHN')                    
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    train_loader, val_loader, test_loader = load_dataset(args.data_dir, download=args.download, batch_size=args.batch_size,dataset=args.dataset)

    # define model
    model = CapsuleNet(input_size=[1, 28, 28], classes=10, routings=3)
    model.cuda()
    #print(model)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))
    if not (args.testing or args.threshold or args.fgsm or args.pgd or args.cwl2 or args.bim):
        train(model, train_loader, test_loader, args)
    if args.threshold:
        print(threshold(model,test_loader,args))
    else:  # testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        if args.testing:
          test_loss, test_acc = test(model=model, test_loader=test_loader,val_loader = test_loader, args=args)
          print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
          show_reconstruction(model, test_loader, 50, args)
        elif args.fgsm:
          test_loss, test_acc = test_fgsm(model=model, test_loader=test_loader, val_loader = test_loader,args=args)
          print('FGSM : Su = %.4f, test loss = %.5f' % (test_acc, test_loss))
          show_reconstruction(model, test_loader, 50, args)
        elif args.pgd:
          test_loss, test_acc = test_pgd(model=model, test_loader=test_loader, val_loader = test_loader, args=args)
          print('PGD : Su = %.4f, test loss = %.5f' % (test_acc, test_loss))
          show_reconstruction(model, test_loader, 50, args)
        elif args.bim:
          test_loss, test_acc = test_bim(model=model, test_loader=test_loader, val_loader  = test_loader, args=args)
          print('BIM : Su = %.4f, test loss = %.5f' % (test_acc, test_loss))
          show_reconstruction(model, test_loader, 50, args)  
        elif args.cwl2:
          test_loss, test_acc = test_cw(model=model, test_loader=test_loader, val_loader = test_loader, args=args)
          print('CW : Su = %.4f, test loss = %.5f' % (test_acc, test_loss))
          show_reconstruction(model, test_loader, 50, args)    