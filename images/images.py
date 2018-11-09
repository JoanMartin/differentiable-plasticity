# Differentiable plasticity: natural image memorization and reconstruction.
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# This program uses the click module rather than argparse to scan command-line arguments. I won't do that again. 

# You start getting acceptable results after ~3000 episodes (~15 minutes with a standard GPU).
# Let it run longer for better results.

# To observe the results, run testpics.py (which uses the output files produced by this program)

import torch
import torch.nn as nn
from torch.autograd import Variable
import click
import numpy as np
from numpy import random
import torch.nn.functional as F
import random
import sys
import pickle
import time
import platform

# Loading the image data. This requires downloading the
# CIFAR 10 dataset (Python version) - https://www.cs.toronto.edu/~kriz/cifar.html
imagedata = np.zeros((0, 1024 * 3))
for numfile in range(4):
    with open('./data_batch_' + str(numfile + 1), 'rb') as fo:
        # imagedict = pickle.load(fo)  # Python 2
        imagedict = pickle.load(fo, encoding='bytes')  # Python 3
    imagedata = np.concatenate((imagedata, imagedict[b'data']), axis=0)

np.set_printoptions(precision=4)

defaultParams = {
    'nbpatterns': 3,  # number of images per episode
    'nbprescycles': 3,  # number of presentations for each image
    'prestime': 20,  # number of time steps for each image presentation
    'prestimetest': 3,  # number of time steps for the test (degraded) image
    'interpresdelay': 2,  # number of time steps (with zero input) between two presentations
    'patternsize': 1024,  # size of the images (32 x 32 = 1024)
    'nbiter': 100000,  # number of episodes

    # when contiguousperturbation is False (which it shouldn't be),
    # probability of zeroing each pixel in the test image
    'probadegrade': .5,

    'lr': 1e-4,  # Adam learning rate
    'print_every': 10,  # how often to print statistics and save files
    'homogenous': 0,  # whether alpha should be shared across connections
    'rngseed': 0  # random seed
}

# ttype = torch.FloatTensor         # For CPU
ttype = torch.cuda.FloatTensor  # For GPU


# Generate the full list of inputs for an episode
def generateInputsAndTarget(params, contiguousperturbation=True):
    inputT = np.zeros((params['nbsteps'], 1, params['nbneur']))  # inputTensor, initially in numpy format...

    # Create the random patterns to be memorized in an episode
    # Floating-point, graded patterns, zero-mean
    patterns = []
    for nump in range(params['nbpatterns']):
        numpic = np.random.randint(imagedata.shape[0])
        p = imagedata[numpic].reshape((3, 1024)).sum(0).astype(float)
        p = p[:params['patternsize']]
        p = p - np.mean(p)
        p = p / (1e-8 + np.max(np.abs(p)))
        # p = (np.random.randint(2, size=params['patternsize']) - .5) *2   # Binary patterns
        patterns.append(p)

    # Now 'patterns' contains the NBPATTERNS patterns to be memorized in this episode - in numpy format
    # Creating the test pattern, partially zero'ed out, that the network will have to complete
    testpattern = random.choice(patterns).copy()
    preservedbits = np.ones(params['patternsize'])

    # Contiguous perturbation = one contiguous half of the image is zeroed out. Default (see above).
    if contiguousperturbation:
        preservedbits[int(params['patternsize'] / 2):] = 0
        if np.random.rand() < .5:
            preservedbits = 1 - preservedbits

    # Otherwise, randomly zero out individual pixels. Because natural images are highly
    # autocorrelated, a trivial approximate solution is to take the average of nearby pixels.
    else:
        preservedbits[:int(params['probadegrade'] * params['patternsize'])] = 0
        np.random.shuffle(preservedbits)
    degradedtestpattern = testpattern * preservedbits

    # Inserting the inputs in the input tensor at the proper places
    for nc in range(params['nbprescycles']):
        np.random.shuffle(patterns)
        for ii in range(params['nbpatterns']):
            for nn in range(params['prestime']):
                numi = nc * (params['nbpatterns'] * (params['prestime'] + params['interpresdelay'])) + \
                       ii * (params['prestime'] + params['interpresdelay']) + nn
                inputT[numi][0][:params['patternsize']] = patterns[ii][:]

    for nn in range(params['prestimetest']):
        inputT[-params['prestimetest'] + nn][0][:params['patternsize']] = degradedtestpattern[:]

    for nn in range(params['nbsteps']):
        inputT[nn][0][-1] = 1.0  # Bias neuron is forced to 1
        # inputT[nn] *= params['inputboost']       # Strengthen inputs

    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    target = torch.from_numpy(testpattern).type(ttype)

    return inputT, target


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()

        # Notice that the vectors are row vectors, and the matrices are transposed
        # wrt the comp neuro order, following deep learning / pytorch conventions
        # Each *column* of w targets a single output neuron
        self.w = Variable(.01 * torch.randn(params['nbneur'], params['nbneur']).type(ttype), requires_grad=True)

        if params['homogenous'] == 1:
            # plasticity coefficients: homogenous/shared across connections
            self.alpha = Variable(.01 * torch.ones(1).type(ttype), requires_grad=True)
        else:
            # plasticity coefficients: independent
            self.alpha = Variable(.01 * torch.randn(params['nbneur'], params['nbneur']).type(ttype),
                                  requires_grad=True)

        # "learning rate" of plasticity, shared across all connections
        self.eta = Variable(.01 * torch.ones(1).type(ttype), requires_grad=True)
        self.params = params

    def forward(self, input, yin, hebb):
        # Inputs are fed by clamping the output of cells that receive
        # input at the input value, like in standard Hopfield networks
        # clamps = torch.zeros(1, self.params['nbneur'])
        clamps = np.zeros(self.params['nbneur'])
        zz = torch.nonzero(input.data[0].cpu()).numpy().squeeze()

        clamps[zz] = 1

        clamps = Variable(torch.from_numpy(clamps).type(ttype), requires_grad=False).float()
        yout = F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb))) * (1 - clamps) + input * clamps

        # bmm used to implement outer product
        hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0]
        return yout, hebb

    def initialZeroState(self):
        return Variable(torch.zeros(1, self.params['nbneur']).type(ttype))

    def initialZeroHebb(self):
        return Variable(torch.zeros(self.params['nbneur'], self.params['nbneur']).type(ttype))


def train(paramdict=None):
    print("Starting training...")
    params = {}
    params.update(defaultParams)
    if paramdict:
        params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    sys.stdout.flush()

    # Total number of steps per episode
    params['nbsteps'] = params['nbprescycles'] * ((params['prestime'] + params['interpresdelay']) *
                                                  params['nbpatterns']) + params['prestimetest']
    params['nbneur'] = params['patternsize'] + 1

    # Turning the parameters into a nice suffix for filenames; rngseed always appears last
    suffix = "images_" + "".join([str(x) + "_"
                                  if pair[0] is not 'nbneur'
                                     and pair[0] is not 'nbsteps'
                                     and pair[0] is not 'print_every'
                                     and pair[0] is not 'rngseed'
                                  else ''
                                  for pair in zip(params.keys(), params.values())
                                  for x in pair])[:-1] + '_rngseed_' + str(params['rngseed'])

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed'])
    random.seed(params['rngseed'])
    torch.manual_seed(params['rngseed'])
    # print(click.get_current_context().params)

    print("Initializing network")
    net = Network(params)
    total_loss = 0.0

    print("Initializing optimizer")
    optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=params['lr'])
    all_losses = []

    nowtime = time.time()
    print("Starting episodes...")
    sys.stdout.flush()

    for numiter in range(params['nbiter']):
        y = net.initialZeroState()
        hebb = net.initialZeroHebb()
        optimizer.zero_grad()

        inputs, target = generateInputsAndTarget(params)

        # Running the episode
        for numstep in range(params['nbsteps']):
            y, hebb = net(Variable(inputs[numstep], requires_grad=False), y, hebb)

        # Computing gradients, applying optimizer
        loss = (y[0][:params['patternsize']] - Variable(target, requires_grad=False)).pow(2).sum()
        loss.backward()
        optimizer.step()

        lossnum = loss.data[0]
        total_loss += lossnum

        # Printing statistics, saving files
        if (numiter + 1) % params['print_every'] == 0:

            print(numiter, "====")
            td = target.cpu().numpy()
            yd = y.data.cpu().numpy()[0][:-1]

            print("y: ", yd[:10])
            print("target: ", td[:10])

            absdiff = np.abs(td - yd)

            print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
            print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])

            previoustime = nowtime
            nowtime = time.time()

            print("Time spent on last", params['print_every'], "iters: ", nowtime - previoustime)

            total_loss /= params['print_every']
            all_losses.append(total_loss)

            print("Mean loss over last", params['print_every'], "iters:", total_loss)
            print("Saving local files...")
            sys.stdout.flush()

            with open('results_' + suffix + '.dat', 'wb') as fo:
                pickle.dump(net.w.data.cpu().numpy(), fo)
                pickle.dump(net.alpha.data.cpu().numpy(), fo)
                pickle.dump(net.eta.data.cpu().numpy(), fo)
                pickle.dump(all_losses, fo)
                pickle.dump(params, fo)

            print("ETA:", net.eta.data.cpu().numpy())
            with open('loss_' + suffix + '.txt', 'w') as thefile:
                for item in all_losses:
                    thefile.write("%s\n" % item)

            sys.stdout.flush()
            sys.stderr.flush()

            total_loss = 0


@click.command()
@click.option('--nbpatterns', default=defaultParams['nbpatterns'])
@click.option('--nbprescycles', default=defaultParams['nbprescycles'])
@click.option('--homogenous', default=defaultParams['prestime'])
@click.option('--prestime', default=defaultParams['prestime'])
@click.option('--prestimetest', default=defaultParams['prestimetest'])
@click.option('--interpresdelay', default=defaultParams['interpresdelay'])
@click.option('--patternsize', default=defaultParams['patternsize'])
@click.option('--nbiter', default=defaultParams['nbiter'])
@click.option('--probadegrade', default=defaultParams['probadegrade'])
@click.option('--lr', default=defaultParams['lr'])
@click.option('--print_every', default=defaultParams['print_every'])
@click.option('--rngseed', default=defaultParams['rngseed'])
def main():
    train(paramdict=dict(click.get_current_context().params))


if __name__ == "__main__":
    main()
