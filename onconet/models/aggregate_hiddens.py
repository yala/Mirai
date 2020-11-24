import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
from onconet.models.factory import RegisterModel

@RegisterModel("all_image_lstm")
class AllImageLSTM(nn.Module):
    def __init__(self, args):
        '''
            Given some a patch model, add add some FC layers and a shortcut to make whole image prediction
       '''
        super(AllImageLSTM, self).__init__()


        args.wrap_model = False
        self.args = args

        self.lstm = nn.LSTM(input_size = args.hidden_dim,
                            hidden_size = args.hidden_dim // 2,
                            num_layers = 1,
                            bias = True,
                            batch_first = True,
                            dropout= args.dropout,
                            bidirectional=True)

        self.view_bn = nn.BatchNorm1d(args.hidden_dim *2)
        self.view_fc = nn.Linear(args.hidden_dim *2, args.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.side_bn = nn.BatchNorm1d(args.hidden_dim *2)
        self.side_fc = nn.Linear(args.hidden_dim *2, args.hidden_dim)
        self.exam_bn = nn.BatchNorm1d(args.hidden_dim *2)
        self.exam_fc = nn.Linear(args.hidden_dim *2, args.hidden_dim)
        self.dropout = nn.Dropout(p=args.dropout)
        self.r_y_fc = nn.Linear(args.hidden_dim, 2)
        self.l_y_fc = nn.Linear(args.hidden_dim, 2)
        self.y_fc = nn.Linear(args.hidden_dim, 2)


    def forward(self, x):
        '''
            param x: a batch of image tensors, in the order of:
                [Cu L CC, Pr L CC, Cu L MLO, Pr L MLO,
                    Cu R CC, Pr R CC, Cu R MLO, Pr R MLO]

            returns hidden: last hidden layer of model
        '''

        # x is shape (B, 8, H). Reshape to  B*4,
        x = self.dropout(x)
        B, N, H = x.size()
        x = x.view(B*4, 2, H)
        # Run each view and it's prior through LSTM then concat
        view_h, _ = self.lstm(x)
        view_h = view_h.contiguous().view(B*4, -1)
        view_h = self.view_bn(view_h)
        side_h = self.relu(self.view_fc(view_h)).contiguous().view(B*2, -1)
        side_h = self.side_bn(side_h)
        exam_h = self.relu(self.side_fc(side_h)).contiguous().view(B,-1)
        exam_h = self.exam_bn(exam_h)
        exam_h = self.dropout(self.relu(self.exam_fc(exam_h)))

        logit, l_logit, r_logit = self.y_fc(exam_h), self.l_y_fc(exam_h), self.r_y_fc(exam_h)

        return logit, l_logit, r_logit, exam_h

@RegisterModel("simpler_all_image_lstm")
class SimplerAllImageLSTM(nn.Module):
    def __init__(self, args):
        '''
            Given some a patch model, add add some FC layers and a shortcut to make whole image prediction
       '''
        super(SimplerAllImageLSTM, self).__init__()


        args.wrap_model = False
        self.args = args

        self.lstm = nn.LSTM(input_size = args.hidden_dim * 4,
                            hidden_size = args.hidden_dim // 2,
                            num_layers = 1,
                            bias = True,
                            batch_first = True,
                            dropout= args.dropout,
                            bidirectional=True)

        self.dropout = nn.Dropout(p=args.dropout)
        self.r_y_fc = nn.Linear(args.hidden_dim * 2, 2)
        self.l_y_fc = nn.Linear(args.hidden_dim * 2, 2)
        self.y_fc = nn.Linear(args.hidden_dim * 2, 2)


    def forward(self, x):
        '''
            param x: a batch of image tensors, in the order of:
                [Cu L CC, Cu L MLO, Cu R CC, Cu R MLO
                    Pr L CC, Pr L MLO, Pr R CC,  Pr R MLO]

            returns hidden: last hidden layer of model
        '''

        # x is shape (B, 8, H). Reshape to  B*4,
        x = self.dropout(x)
        B, N, H = x.size()
        x = x.view(B, 2, H*4)
        # Run each view and it's prior through LSTM then concat
        h, _ = self.lstm(x)
        h = h.contiguous().view(B, -1)
        h = self.dropout(h)
        logit, l_logit, r_logit = self.y_fc(h), self.l_y_fc(h), self.r_y_fc(h)

        return logit, l_logit, r_logit, h


@RegisterModel("all_cur_image_fc")
class AllCurImageFC(nn.Module):
    def __init__(self, args):
        '''
            Given some a patch model, add add some FC layers and a shortcut to make whole image prediction
       '''
        super(AllCurImageFC, self).__init__()


        args.wrap_model = False
        self.args = args

        self.dropout = nn.Dropout(p=args.dropout)
        self.r_y_fc = nn.Linear(args.hidden_dim * 4, 2)
        self.l_y_fc = nn.Linear(args.hidden_dim * 4, 2)
        self.y_fc = nn.Linear(args.hidden_dim * 4, 2)


    def forward(self, x):
        '''
            param x: a batch of image tensors, in the order of:
                [Cu L CC, Cu L MLO, Cu R CC, Cu R MLO
                    Pr L CC, Pr L MLO, Pr R CC,  Pr R MLO]

            returns hidden: last hidden layer of model
        '''

        # x is shape (B, 8, H). Reshape to  B*4,
        x = self.dropout(x)
        B, N, H = x.size()
        x = x.view(B, -1)
        logit, l_logit, r_logit = self.y_fc(x), self.l_y_fc(x), self.r_y_fc(x)
        return logit, l_logit, r_logit, x
