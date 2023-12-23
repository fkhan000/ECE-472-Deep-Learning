import tensorflow as tf
from Linear import Linear
import numpy as np
from Norm import Norm_Layer
from MLP import MLP

#This is our multihead attention block
class MultiHeadAttention(tf.Module):

    def __init__(self, d_model, num_heads, causal_mask = False):

        #we store the projection matrices for the query, key, and value
        #vectors in a list of tuples of Linear objects
        self.proj_matrices = []
        self.causal_mask = causal_mask

        #for each head
        for index in range(num_heads):
            #we append a set of projection matrices. Since we want to split
            #up the input evenly among the heads our projection matrix
            #will project the d_model vectors to a smaller dimension
            self.proj_matrices += [
                Linear(d_model, d_model // num_heads, bias = False),
                Linear(d_model, d_model // num_heads, bias = False),
                Linear(d_model, d_model // num_heads, bias = False)
                ]
        
        #to make sure that the query, key and value vectors become "specialized"
        #(the model learns what their purpose is), we feed copies of the input
        #to three different projection matricies whose outputs become our
        #query, key, and value vectors
        self.FC = []

        for index in range(3):
            self.FC.append(Linear(d_model, d_model))
        
        #Finally, after concatenating the outputs from each of the heads
        #we feed it into a linear layer
        self.output_layer = Linear(d_model, d_model, bias = False)

    #This function performs the scaled dot product attention
    def attention(self, Q, K, V):

        #we calculate QK^T and then scale it down by the size of the query vector
        b = tf.einsum("bij,bkj->bik", Q, K) / np.sqrt(Q.shape[1])

        #if we're using tihs block in a decoder
        #then we will need to apply a causal mask to the input
        #so as to ensure that the model doesn't cheat and learn
        #future inputs
        if self.causal_mask:
            #we save the diagonal of b
             diag = tf.linalg.diag_part(b)
             #create our "infinity mask". When I used negative infinity
             #it ended up breaking the gradient flow so I decided to use
             #a really large negative number instead
             inf_mask = tf.fill(b.shape, -1e9)

             #we set the upper half of our matrix to negative infinity
             b = b +  tf.linalg.band_part(inf_mask, 0, -1)

            #and then because the previous operation was applied to the diagonal
             #as well, we retrieve back the diagonal
             b = tf.linalg.set_diag(b, diag)

        #we apply softmax
        b = tf.nn.softmax(b)

        #and then multiply our matrix by values
        return b @ V

    
    #This function produces the output of our attention block
    def __call__(self, x):

        #we feed the input through the three linear layers to get
        #our query, key, and value vectors
        Q = self.FC[0](x)
        K = self.FC[1](x)
        V = self.FC[2](x)        
        
        output = []
        index = 0
        #then for each head
        while(index < len(self.proj_matrices)):

            #we calculate the attention matrix
            att_score = self.attention(self.proj_matrices[index](Q),
                                       self.proj_matrices[index+1](K),
                                       self.proj_matrices[index+2](V))
            #append to our output list
            output.append(att_score)
            index += 3
            
        #and then concatenate all of the outputs from the heads
        return self.output_layer(tf.concat(output, axis = -1))



#This class implements our transformer block
class Transformer(tf.Module):

    def __init__(self, d_model, num_heads, num_blocks, causal_mask = False):

        self.d_model = d_model
        self.norm_layers = []
        #we store each of the mha blocks in a list
        self.blocks = []

        self.causal_mask = causal_mask
        for index in range(num_blocks):

            
            self.blocks.append(MultiHeadAttention(d_model, num_heads,
                                                  causal_mask = self.causal_mask))
            #as well as the normalization layers
            self.norm_layers.append(Norm_Layer())

        self.norm_layers.append(Norm_Layer())
        
        self.FFN = MLP(d_model, d_model,
                       1, 2048,
                       hidden_activation = tf.nn.relu)

        #in order to ensure that our model learns the order of the words
        #we add positional encodings to the input matrix
        self.pos_encodings = tf.math.sin(tf.constant(np.arange(d_model), dtype = tf.float32)/(10000**(2/d_model)
                                                                         ))
        

    #This function produces the output of our model
    def __call__(self, x):

        #we add the encodings to the input
        z = self.pos_encodings +  x

        #for each mha block
        for index in range(len(self.blocks)):

            #since every block has a skip connection,
            #that bypasses it, we add the output of the mha block
            #to our pending output
            z += self.blocks[index](z)

            #feed it into the normalization layer
            z = self.norm_layers[index](z)

        #send it through an FFN
        z += self.FFN(z)

        #and then normalize it one last time
        z = self.norm_layers[-1](z)

        return z

            


 

        
    
        
        
        
