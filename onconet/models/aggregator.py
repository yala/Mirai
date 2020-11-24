import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
from onconet.models.factory import RegisterModel, strip_model, get_output_size

@RegisterModel("aggregator")
class Aggregator(nn.Module):
    def __init__(self, args):
        '''
            Given some a patch model, add add some FC layers and a shortcut to make whole image prediction
       '''
        super(Aggregator, self).__init__()
        print('\nLoading patch model from [%s]...' % args.patch_snapshot)
        try:
            patch_model = torch.load( args.patch_snapshot).cpu()
            self.patch_model = strip_model( patch_model )
        except Exception as e:
            raise Exception("Couldn't load patch model at {}. Error: {}".format(args.patch_snapshot, e));

        args.wrap_model = False
        self.args = args

        if args.multi_image:
            img_size = ( args.num_images, *args.img_size)

        args.hidden_dim = get_output_size(self.patch_model , img_size, args.num_chan, args.cuda)
        fc1_dim = max(2056, args.hidden_dim/8)
        fc2_dim = max(1024, args.hidden_dim/16)

        self.fc1 = nn.Linear( args.hidden_dim, fc1_dim)
        self.fc2 = nn.Linear( fc1_dim, fc2_dim)
        self.fc_final = nn.Linear( fc2_dim, args.num_classes)


    def forward(self, x):
        '''
            param x: a batch of image tensors
            returns hidden: last hidden layer of model (as if wrapper wasn't applied)
        '''

        ## We assume path model is wrapped in a modelWrapper from a prev run
        patch_hidden  = self.patch_model(x)
        patch_hidden = patch_hidden.view(patch_hidden.size()[0], -1)
        patch_hidden = F.relu( self.fc1( patch_hidden) )
        hidden = F.relu(self.fc2(patch_hidden))
        logit = self.fc_final( hidden )
        return logit
