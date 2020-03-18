#!/usr/bin/env python3
# coding: utf-8

# Continue training an already initialized model, but adjust output to be kcal/mol
# In[1]:


import pyanitools as pya
import molgrid
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os, glob
import matplotlib.pyplot as plt
import wandb
import argparse



parser = argparse.ArgumentParser(description='Progressively train on ANI data (which is in current directory)')

parser.add_argument("--maxepoch",default=100,type=int,help="Number of epochs before moving on")
parser.add_argument("--stop",default=20000, type=int, help="Number of iterations without improvement before moving on")
parser.add_argument("--lr",default=0.001,type=float, help="Initial learning rate")
parser.add_argument("--model",required=True,type=str, help="pretrained model file")
parser.add_argument("--normalize",action='store_true',help="Normalize energies by number of atoms")
args = parser.parse_args()

typemap = {'H': 0, 'C': 1, 'N': 2, 'O': 3} #type indices
typeradii = [1.0, 1.6, 1.5, 1.4] #not really sure how sensitive the model is to radii





#this is Daniela's model
class Net(nn.Module):
    def __init__(self, dims):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(dims[0], 64, kernel_size=5, padding=2)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=5, padding=2)
        self.conv2b =nn.Conv3d(128,128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(2)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.last_layer_size = (dims[1]//16)**3 * 256
        self.fc1 = nn.Linear(self.last_layer_size, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv1b(x))
        x = self.pool1(x)
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv2b(x))
        x = self.pool2(x)
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = F.elu(self.conv4(x))
        x = self.pool4(x)
        x = F.elu(self.conv5(x))

        x = x.view(-1, self.last_layer_size)
        x = self.fc1(x)
        return x.flatten()

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)


gmaker = molgrid.GridMaker(resolution=0.25, dimension = 15.75)
batch_size = 25
dims = gmaker.grid_dimensions(4) # 4 types
tensor_shape = (batch_size,)+dims  #shape of batched input



#allocate tensors
input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
labels = torch.zeros(batch_size, dtype=torch.float32, device='cuda')


TRAIL = 40

def train_strata(strata, model, optimizer, losses, maxepoch, stop=20000, initloss=1000):
    bestindex = len(losses) #position    
    bestloss=100000
    for _ in range(maxepoch):  #do at most MAXEPOCH epochs, but should bail earlier
        np.random.shuffle(strata)
        for pos in range(0,len(strata),batch_size):
            batch = strata[pos:pos+batch_size]
            if len(batch) < batch_size: #wrap last batch
                batch += strata[:batch_size-len(batch)]
            batch = molgrid.ExampleVec(batch)
            batch.extract_label(0,labels) # extract first label (there is only one in this case)

            gmaker.forward(batch, input_tensor, 2, random_rotation=True)  #create grid; randomly translate/rotate molecule
            output = model(input_tensor) #run model
            loss = F.smooth_l1_loss(output,labels)  # THIS PART DIFFERENT
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),10)

            optimizer.step()
            losses.append(float(loss))
            trailing = np.mean(losses[-TRAIL:])
            
            if trailing < bestloss:
                bestloss = trailing
                bestindex = len(losses)
            
            wandb.log({'loss': float(loss),'trailing':trailing,'bestloss':bestloss,'stratasize':len(strata),'lr':optimizer.param_groups[0]['lr']})
            
            if len(losses)-bestindex > stop and bestloss < initloss:
                return bestloss # "converged"
    return bestloss



wandb.init(project="ani", config=args)




losses = []
model = Net(dims).to('cuda')
model.load_state_dict(torch.load(args.model))
optimizer = optim.SGD(model.parameters(), lr=args.lr)    
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


# load the data into memory (big)


mcnt = 0 #molecules
ccnt = 0 #conformers
elements = set()
examples = [] #the entire training set loaded into memory
examplesbysize = dict()
for hd5file in sorted(glob.glob('*.h5')):
    for data in pya.anidataloader(hd5file):
        #calculate some statistics
        mcnt += 1
        ccnt += len(data['energies'])
        elements.update(data['species'])
        
        #molecule types and radii
        types = np.array([typemap[elem] for elem in data['species']], dtype=np.float32)
        radii = np.array([typeradii[int(index)] for index in types], dtype=np.float32)

        sz = len(radii)
        if sz not in examplesbysize:
            examplesbysize[sz] = []
        #create an example for every conformer
        for coord, energy in zip(data['coordinates'],data['energies']):
            c = molgrid.CoordinateSet(coord.astype(np.float32), types, radii,4)
            ex = molgrid.Example()
            ex.coord_sets.append(c)
            energy *= 627.5096 #convert to kcal/mol
            
            if args.normalize:
                energy /= sz
            ex.labels.append(energy)        
            examples.append(ex)
            examplesbysize[sz].append(ex)
    

wandb.watch(model)

for sz in range(2,27):
    #construct strata of molecules with <= sz atoms
    strata = []
    for i in range(2,sz):
        strata += examplesbysize[i]
                
#training on the full training set, start stepping the learning rate
bestloss = 100000
for i in range(3):
    bestloss = train_strata(strata, model, optimizer, losses, args.maxepoch, args.stop, bestloss)
    print("Best loss from strata: ",bestloss)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_refine%d.pt'%i))
    scheduler.step()





