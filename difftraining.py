#!/usr/bin/env python3
# coding: utf-8

# Train to differences from linear model


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
import argparse, pickle
import itertools
import math
from se3cnn.image.gated_block import GatedBlock
from SE3ResNet import LargeNetwork, SmallNetwork

parser = argparse.ArgumentParser(description='Progressively train on ANI data (which is in current directory)')

parser.add_argument("--maxepoch",default=100,type=int,help="Number of epochs before moving on")
parser.add_argument("--stop",default=20000, type=int, help="Number of iterations without improvement before moving on")
parser.add_argument("--lr",default=0.001,type=float, help="Initial learning rate")
parser.add_argument("--resolution",default=0.25,type=float, help="Grid resolution")
parser.add_argument("--clip",default=10,type=float, help="Gradient clipping")
parser.add_argument("--solver",default="adam",choices=('adam','sgd'),type=str, help="solver to use (adam|sgd)")
parser.add_argument("--pickle",default="traintest.pickle",type=str)
parser.add_argument("--molcache",default="traintest.molcache2",type=str)
parser.add_argument("--train_types",default="train.types",type=str)
parser.add_argument("--test_types",default="test.types",type=str)
parser.add_argument("--model_type",choices=['ResNet34','Custom'],default='ResNet34',help='Type of model to use')
parser.add_argument("--resnet_type",choices=['Small','Large'],default='Large',help='Type of ResNet34 to use from SE3CNN paper')
parser.add_argument("--num_modules",default=5,type=int,help="number of convolutional modules")
parser.add_argument("--module_depth",default=1,type=int,help="number of layers in module")
parser.add_argument("--module_connect",default="straight",choices=('straight','dense','residual'),type=str, help="how module is connected")
parser.add_argument("--kernel_size",default=3,type=int,help="kernel size of module")
parser.add_argument("--module_filters",default=[64],nargs='+',type=int,help="number of filters in each module")
parser.add_argument("--module_filters_0",type=int,help="number of filters in each module, scalar")
parser.add_argument("--module_filters_1",type=int,help="number of filters in each module, vect")
parser.add_argument("--module_filters_2",type=int,help="number of filters in each module, mult2")
parser.add_argument("--module_filters_3",type=int,help="number of filters in each module, mult3")
parser.add_argument("--filter_factor",default=[2],nargs='+',type=float,help="set filters to this raised to the current module index")
parser.add_argument("--activation_function",default="elu",choices=('elu','relu','sigmoid'),help='activation function')
parser.add_argument("--hidden_size",default=0,type=int,help='size of hidden layer, zero means none')
parser.add_argument("--pool_type",default="max",choices=('max','ave'),help='type of pool to use between modules')
parser.add_argument("--conv_type",default="conv",choices=('conv','se3'),help='type of convolution to use, "conv" is normal nn.Conv3d and "se3" is equivariant convolution')
parser.add_argument("--final_scalar",default=0,choices=(0,1),help='use a final SE3CNN for producing only scalar values')

#for ResNet34 models
parser.add_argument("--p-drop-conv", type=float, default=None,
                        help="convolution/capsule dropout probability")
parser.add_argument("--p-drop-fully", type=float, default=None,
                    help="fully connected layer dropout probability")
parser.add_argument("--bandlimit-mode", choices={"conservative", "compromise", "sfcnn"}, default="compromise",
                    help="bandlimiting heuristic for spherical harmonics")
parser.add_argument("--SE3-nonlinearity", choices={"gated", "norm"}, default="gated",
                    help="Which nonlinearity to use for non-scalar capsules")
parser.add_argument("--normalization", choices={'batch', 'group', 'instance', None}, default='batch',
                    help="Which nonlinearity to use for non-scalar capsules")
parser.add_argument("--downsample-by-pooling", action='store_true', default=True,
                    help="Switches from downsampling by striding to downsampling by pooling")
args = parser.parse_args()

if args.model_type == 'Custom':
    if args.module_filters_1 or args.module_filters_0 or args.module_filters_2 or args.module_filters_3:
        args.module_filters = tuple([val if val else 0 for val in [args.module_filters_0, args.module_filters_1, args.module_filters_2, args.module_filters_3] ])
        print(args.module_filters)

typemap = {'H': 0, 'C': 1, 'N': 2, 'O': 3} #type indices
typeradii = [1.0, 1.6, 1.5, 1.4] #not really sure how sensitive the model is to radii


#load data
batch_size = 16
if args.module_connect == 'dense':
    batch_size = 8
typer = molgrid.SubsettedGninaTyper([0,2,6,10],catchall=False)
examples = molgrid.ExampleProvider(typer,molgrid.NullIndexTyper(),recmolcache=args.molcache, shuffle=True,default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
examples.populate(args.train_types)
valexamples = molgrid.ExampleProvider(typer,molgrid.NullIndexTyper(),recmolcache=args.molcache,default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
valexamples.populate(args.test_types)
# (train, test) = pickle.load(open(args.pickle,'rb'))

# def load_examples(T):
#     examples = []
#     for coord, types, energy, diff in T:
#         radii = np.array([typeradii[int(index)] for index in types], dtype=np.float32)
#         c = molgrid.CoordinateSet(coord, types, radii,4)
#         ex = molgrid.Example()
#         ex.coord_sets.append(c)
#         ex.labels.append(diff)        
#         examples.append(ex)
#     return examples
  
# examples = load_examples(train)
# del train
# valexamples = load_examples(test)
# del test
  

class View(nn.Module):
    def __init__(self, shape):        
        super(View, self).__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(*self.shape)
        
class Net(nn.Module):
    def __init__(self, dims):
        super(Net, self).__init__()
        self.modules = []
        self.residuals = []
        nchannels = dims[0]
        dim = dims[1]
        ksize = args.module_kernel_size
        pad = ksize//2
        fmult = 1
        func = F.elu
        if args.activation_function == 'relu':
            func = F.relu
        elif args.activation_function == 'sigmoid':
            func = F.sigmoid

        pooler = nn.MaxPool3d
        if args.pool_type == 'avg':
            pooler = nn.AvgPool3d

        if args.conv_type == 'se3':
            if isinstance(nchannels, int):
                nchannels = (nchannels, )
            fmult = (1, )
        elif args.conv_type == 'conv':
            args.module_filters = args.module_filters[0]
            args.filter_factor = args.filter_factor[0]
            assert isinstance(nchannels, int) and isinstance(args.module_filters, int)

            
        inmultincr = 0
        if args.module_connect == 'dense':
            inmultincr = 1
            if args.conv_type == 'conv':
                assert isinstance(nchannels, int) and isinstance(args.module_filters, int)
                conv = nn.Conv3d(nchannels, args.module_filters, kernel_size=ksize, padding=pad)
            elif args.conv_type == 'se3':
                activ = self.getGatedBlockActivation(nchannels,args.module_filters)
                conv = GatedBlock(nchannels, args.module_filters, size=ksize, padding=pad, stride=1, activation=activ)
            self.add_module('init_conv', conv)
            self.modules.append([conv,func])
            nchannels = args.module_filters
            
        for m in range(args.num_modules):
            module = []          
            # inmult = 1
            if args.conv_type == 'se3':
                filters = tuple([int(mfilt*fm) for mfilt, fm in itertools.zip_longest(args.module_filters,fmult,fillvalue=0)])
            else:
                filters = int(args.module_filters*fmult  )
            startchannels = nchannels
            for i in range(args.module_depth):
                if args.conv_type == 'conv':
                    assert isinstance(nchannels, int)
                    conv = nn.Conv3d((1-inmultincr)*nchannels+inmultincr*(startchannels+(i*filters)), filters, kernel_size=ksize, padding=pad)
                elif args.conv_type == 'se3':
                    in_nchannels = tuple([(1-inmultincr)*chan + inmultincr*(s_nchan+(i*filt))
                        for chan, s_nchan, filt in itertools.zip_longest(nchannels, startchannels, filters, fillvalue=0)])
                    activ = self.getGatedBlockActivation(in_nchannels,filters)
                    conv = GatedBlock(in_nchannels, filters, size=ksize, padding=pad, stride=1, activation=activ)

                # inmult += inmultincr
                self.add_module('conv_%d_%d'%(m,i), conv)
                module.append(conv)
                if args.conv_type == 'conv':
                    module.append(func)
                else: ## don't need activation if SE3 (already included in GatedBlock)
                    module.append(nn.Identity())
                nchannels = filters
            
            if args.module_connect == 'residual':
                #create a 1x1x1 convolution to match input filters to output
                if args.conv_type == 'conv':
                    conv = nn.Conv3d(startchannels, nchannels, kernel_size=1, padding=0)
                elif args.conv_type == 'se3':
                    activ = self.getGatedBlockActivation(startchannels,nchannels)
                    conv = GatedBlock(startchannels, nchannels, size=1, padding=0, stride=1, activation=activ)
                    
                self.add_module('resconv_%d'%m,conv)
                self.residuals.append(conv)
            #don't pool on last module
            if m < args.num_modules-1:
                pool = pooler(2)
                self.add_module('pool_%d'%m,pool)
                module.append(pool)
                dim /= 2
            self.modules.append(module)
            if args.conv_type == 'conv':
                fmult *= args.filter_factor
            else:
                fmult = tuple([fm * ff for fm, ff in itertools.zip_longest(fmult,args.filter_factor,fillvalue=0)])
            
        lastmod = []
        last_size = dim**3
        if args.conv_type == 'conv':
            last_size = int(last_size * filters)
        else:
            if args.final_scalar == 1:
                filters = tuple([torch.sum(filters),0,0,0])
                print(f"final_filters:{filters}")
                conv = GatedBlock(in_nchannels, filters, size=ksize, padding=pad, stride=1, activation=activ) 
                self.add_module('scalar_conv',conv)
                lastmod.append(conv)
            last_size = int(sum([(2*l+1)*mult for l, mult in enumerate(filters)])*last_size)
        lastmod.append(View((-1,last_size)))
        
        if args.hidden_size > 0:
            fc = nn.Linear(last_size, args.hidden_size)
            self.add_module('hidden',fc)
            lastmod.append(fc)
            lastmod.append(func)
            last_size = args.hidden_size
            
        fc = nn.Linear(last_size, 1)
        self.add_module('fc',fc)
        lastmod.append(fc)
        lastmod.append(nn.Flatten())
        self.modules.append(lastmod)
            

    def decomposeTensorMultiplicities(self, x, mult, dims, gate_act):
        end = None
        if sum(mult[1:]) and gate_act: #there is a scalar gate layer that doesn't affect output size but only if there are non scalars
            end = -1
        splits = [m * l for m, l in zip(mult[:end], dims[:end])]
        
        return x.split(splits,dim=1)

    def forward(self, x):
        isdense = False
        isres = False
        if args.module_connect == 'dense':
            isdense = True
        if args.module_connect == 'residual':
            isres = True
                        
        for (m,module) in enumerate(self.modules):
            prevconvs = [x]
            last_mult, last_dims, last_ga = None, None, None
            if isres and len(self.residuals) > m:
                #apply convolution
                passthrough = self.residuals[m](x)
                if isinstance(self.residuals[m], GatedBlock):
                    last_mult = self.residuals[m].conv.kernel.multiplicities_out
                    last_dims = self.residuals[m].conv.kernel.dims_out
                    last_ga = self.residuals[m].gate_act
                    passthrough = self.decomposeTensorMultiplicities(passthrough, last_mult,
                            last_dims, last_ga)
            else:
                isres = False
            for (l,layer) in enumerate(module):
                if isinstance(layer, nn.Conv3d) and isdense:
                    if len(prevconvs) > 1:
                        # concate along channels
                        x = torch.cat(prevconvs,dim=1)
                elif isinstance(layer, GatedBlock) and isdense:
                    if l:
                        x = torch.cat([multiplicity for multiplicity in itertools.chain(*itertools.zip_longest(in_div, out_div))
                            if multiplicity is not None], dim=1)
                    in_div = self.decomposeTensorMultiplicities(x, layer.conv.kernel.multiplicities_in, layer.conv.kernel.dims_in, None)
                if isres and l == len(module)-1:
                    #at last relu, do addition before
                    if isinstance(passthrough, torch.Tensor):
                        x = x + passthrough
                    else:
                        x = self.decomposeTensorMultiplicities(x, last_mult, last_dims, last_ga)
                        x = torch.cat([x_i + p for x_i, p in itertools.zip_longest(x, passthrough, fillvalue=0)],dim=1)

                x = layer(x)
                
                if isdense:
                    if isinstance(layer,GatedBlock):
                        last_mult = layer.conv.kernel.multiplicities_out
                        last_dims = layer.conv.kernel.dims_out
                        last_ga = layer.gate_act
                        out_div = self.decomposeTensorMultiplicities(x, last_mult, last_dims, last_ga)
                    elif isinstance(layer, nn.Conv3d) and isdense:
                        prevconvs.append(x) #save for later

        return x
    
    def getGatedBlockActivation(self,inchannels,outchannels):
        activ_1 = F.relu
        if len(inchannels) == 1 or sum(inchannels[1:]) <= 0:
                activ_1 = None
        activ_2 = F.sigmoid
        if len(outchannels) == 1 or sum(outchannels[1:]) <= 0:
            activ_2 = None

        return (activ_1,activ_2)

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)
    if isinstance(m, GatedBlock):
        # Need to calculate fan_in and fan_out myself, weights are kept as a 1D array normally
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.conv.kernel.forward())
        gain = 1.0
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        init._no_grad_uniform_(m.conv.kernel.weight.data,-a,a)

gmaker = molgrid.GridMaker(resolution=args.resolution, dimension = 16-args.resolution)
dims = gmaker.grid_dimensions(4) # 4 types
tensor_shape = (batch_size,)+dims  #shape of batched input


#allocate tensors
input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
labels = torch.zeros(batch_size, dtype=torch.float32, device='cuda')


TRAIL = 100

# def train_strata(strata, model, optimizer, losses, maxepoch, stop=20000):
#     bestloss = 100000 #best trailing average loss we've seen so far in this strata
#     bestindex = len(losses) #position    
#     for _ in range(maxepoch):  #do at most MAXEPOCH epochs, but should bail earlier
#         np.random.shuffle(strata)
#         for pos in range(0,len(strata),batch_size):
#             batch = strata[pos:pos+batch_size]
#             if len(batch) < batch_size: #wrap last batch
#                 batch += strata[:batch_size-len(batch)]
#             batch = molgrid.ExampleVec(batch)
#             batch.extract_label(0,labels) # extract first label (there is only one in this case)

#             gmaker.forward(batch, input_tensor, 2, random_rotation=True)  #create grid; randomly translate/rotate molecule
#             output = model(input_tensor) #run model
#             loss = F.smooth_l1_loss(output.flatten(),labels.flatten())
#             loss.backward()
            
#             if args.clip > 0:
#               nn.utils.clip_grad_norm_(model.parameters(),args.clip)

#             optimizer.step()
#             losses.append(float(loss))
#             trailing = np.mean(losses[-TRAIL:])
            
#             if trailing < bestloss:
#                 bestloss = trailing
#                 bestindex = len(losses)
#                 torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_better_%d_%d_%f.pt'%(_,pos,bestloss)))
#             if (pos % 100) == 0: 
#                 wandb.log({'loss': float(loss),'trailing':trailing,'bestloss':bestloss,'stratasize':len(strata),'lr':optimizer.param_groups[0]['lr']})
            
#             if len(losses)-bestindex > stop:
#                 return True # "converged"
#     return False

def train_strata(strata, model, optimizer, losses, maxepoch, stop=20000):
    bestloss = 100000 #best trailing average loss we've seen so far in this strata
    bestindex = len(losses) #position    
    for _ in range(maxepoch):  #do at most MAXEPOCH epochs, but should bail earlier
        for pos, batch in enumerate(strata):
            # batch = strata[pos:pos+batch_size]
            # if len(batch) < batch_size: #wrap last batch
            #     batch += strata[:batch_size-len(batch)]
            # batch = molgrid.ExampleVec(batch)
            batch.extract_label(1,labels) # extract second label 

            gmaker.forward(batch, input_tensor, 2, random_rotation=True)  #create grid; randomly translate/rotate molecule
            output = model(input_tensor) #run model
            loss = F.smooth_l1_loss(output.flatten(),labels.flatten())
            loss.backward()
            
            if args.clip > 0:
              nn.utils.clip_grad_norm_(model.parameters(),args.clip)

            optimizer.step()
            losses.append(float(loss))
            trailing = np.mean(losses[-TRAIL:])
            
            if trailing < bestloss:
                bestloss = trailing
                bestindex = len(losses)
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_better_%d_%d_%f.pt'%(_,pos,bestloss)))
            if (pos % 100) == 0: 
                wandb.log({'loss': float(loss),'trailing':trailing,'bestloss':bestloss,'stratasize':strata.large_epoch_size(),'lr':optimizer.param_groups[0]['lr']})
            
            if len(losses)-bestindex > stop:
                return True # "converged"
    return False


# def test_strata(valexamples, model):
#     with torch.no_grad():
#         model.eval()
#         results = []
#         labels = []
#         labelvec = torch.zeros(batch_size, dtype=torch.float32, device='cuda')
#         for pos in range(0,len(valexamples),batch_size):
#             batch = valexamples[pos:pos+batch_size]
#             if len(batch) < batch_size: #wrap last batch
#                 batch += valexamples[:batch_size-len(batch)]
#             batch = molgrid.ExampleVec(batch)
#             batch.extract_label(0,labelvec) # extract first label (there is only one in this case)

#             gmaker.forward(batch, input_tensor, 2, random_rotation=True)  #create grid; randomly translate/rotate molecule
#             output = model(input_tensor)   
#             results.append(output.detach().cpu().numpy())
#             labels.append(labelvec.detach().cpu().numpy())
            
#         results = np.array(results).flatten()
#         labels = np.array(labels).flatten()
#         valrmse = np.sqrt(np.mean((results - labels)**2))
#         if np.isinf(valrmse):
#             valrmse = 1000
#         valame = np.mean(np.abs(results-labels))
#         print("Validation",valrmse,valame)
#         wandb.log({'valrmse': valrmse,'valame':valame})
#         wandb.log({'valpred':results,'valtrue':labels})

def test_strata(valexamples, model):
    with torch.no_grad():
        model.eval()
        results = []
        labels = []
        labelvec = torch.zeros(batch_size, dtype=torch.float32, device='cuda')
        # for pos in range(0,len(valexamples),batch_size):
        for idx, batch in enumerate(valexamples):
#            batch = valexamples[pos:pos+batch_size]
#            if len(batch) < batch_size: #wrap last batch
#                batch += valexamples[:batch_size-len(batch)]
#            batch = molgrid.ExampleVec(batch)
            batch.extract_label(1,labelvec) # extract second label

            gmaker.forward(batch, input_tensor, 2, random_rotation=True)  #create grid; randomly translate/rotate molecule
            output = model(input_tensor)   
            results.append(output.detach().cpu().numpy())
            labels.append(labelvec.detach().cpu().numpy())
            if idx % 1000 == 0:
                print(idx)
            
        results = np.array(results).flatten()
        labels = np.array(labels).flatten()
        valrmse = np.sqrt(np.mean((results - labels)**2))
        if np.isinf(valrmse):
            valrmse = 1000
        valame = np.mean(np.abs(results-labels))
        print("Validation",valrmse,valame)
        wandb.log({'valrmse': valrmse,'valame':valame})
        wandb.log({'valpred':results,'valtrue':labels})
        
          
wandb.init(project="anidiff", config=args)

losses = []
if args.model_type == 'Custom':
    Network = Net(dims)
elif args.model_type == 'ResNet34':
    if args.resnet_type == 'Small':
        Network = SmallNetwork(dims[0],1,args)
    elif args.resnet_type == 'Large':
        Network = LargeNetwork(dims[0],1,args)
model = Network.to('cuda')
model.apply(weights_init)

if args.solver == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


wandb.watch(model)

print("start testing")
test_strata(valexamples, model)
print("done testing")
                
#train on full training set, start stepping the learning rate
for i in range(3):
    train_strata(examples, model, optimizer, losses, args.maxepoch, args.stop)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_refine%d.pt'%i))
    scheduler.step()
    test_strata(valexamples, model)





