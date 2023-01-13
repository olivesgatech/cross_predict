import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class VLSTMModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(VLSTMModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
#         self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.gru = args.gru


        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        #self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        #frame_data = [torch.squeeze(input_, [0]) for input_ in torch.split(0, self.seq_length, input_data)]
        
        #print("***************************")
        #print("input data")
        # Construct the output variable
        input_data = args[0]
        hidden_states = args[1]
        cell_states = args[2]

        if self.gru:
            cell_states = None

        PedsList = args[3]
        num_pedlist = args[4]
        dataloader = args[5]
        look_up = args[6]

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            # Peds present in the current frame

            #print("now processing: %s base frame number: %s, in-frame: %s"%(dataloader.get_test_file_name(), dataloader.frame_pointer, framenum))
            #print("list of nodes")

            nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            #print(PedsList)

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue


            # List of nodes
            #print("lookup table :%s"% look_up)
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:            
                outputs = outputs.cuda()
            #print("list of nodes: %s"%nodeIDs)
            #print("trans: %s"%corr_index)
            #if self.use_cuda:
             #   list_of_nodes = list_of_nodes.cuda()


            #print(list_of_nodes.data)
            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks

            


            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)


            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))

            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(input_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(input_embedded, (hidden_states_current))


            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
    
    
def parse_args():        
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')

    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')

    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')

    parser.add_argument('--num_validation', type=int, default=3,
                        help='Total number of validation dataset for validate accuracy')

    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')

    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')

    
    args = parser.parse_args()
    return args


if __name__=='__main__':
    
    model = VLSTMModel()
    
    import pdb
    pdb.set_trace()
