# Assumed bias is decomposed by weights altogether



import tensorflow as tf



from tensorflow import keras



##############################################################################

##############################################################################

##############################################################################



class TTFC(tf.keras.layers.Layer): 

    def __init__(self, num_outputs, rank_list, dim_list, bias_on=True, activation=None): 

        #rank_list is for dim 1 to dime n-1 but dim list determines the size of dim 1 to dim n and rank 0 and rank n set 1 automatically

        super(TTFC, self).__init__()

        self.num_outputs = num_outputs

        self.rank_list=rank_list

        self.dim_list=dim_list

        self.bias_on=bias_on

        self.rank_list.insert(0,1)

        self.rank_list.insert(len(self.rank_list),1)

        self.dim=len(dim_list)

        

        if activation==None:

            self.activation=tf.keras.activations.linear

        if activation=="sigmoid":

            self.activation=tf.math.sigmoid

        if activation=="relu":

            self.activation=tf.nn.relu

        if activation=="softmax":

            self.activation=tf.nn.softmax

        

        

        

        #self.name=name

    

    def build(self,input_shape):

        

        self.shape_of_input=input_shape

        

       

            

        self.factors=[]

        self.tensor_element_count=1

        for i in range(0,self.dim):

            factor = self.add_weight("factors",

                                  shape=[self.rank_list[i],self.dim_list[i],self.rank_list[i+1]])

       

            self.factors.append(factor)

            self.tensor_element_count=self.dim_list[i]*self.tensor_element_count

        

        self.height=int(self.tensor_element_count/self.num_outputs)

        

        

    def call(self, input_): 

        tensor_kernel=self.factors[0]

        #self.add_loss(tf.math.log(tf.math.square(tf.norm(self.factors[0])))/3000)

        for j in range(len(self.factors)-1):

            

            tensor_kernel=tf.tensordot(tensor_kernel,self.factors[j+1],axes=1)

            

            #self.add_loss(tf.math.log(tf.math.suare(tf.norm(self.factors[j+1])))/300)

            

        if self.bias_on:

            self.kernel_dim=self.shape_of_input[-1]

            kernel=tf.reshape(tensor_kernel,shape=[self.height,self.num_outputs]) #add the bias as a row

            w_kernel=kernel[:self.kernel_dim,:]

            b_kernel=kernel[self.kernel_dim,:] #bias is the last row

            partial_out=tf.matmul(input_,w_kernel)

            output=tf.nn.bias_add(partial_out,b_kernel)

            

        else:

            self.kernel_dim=self.shape_of_input[-1]

            kernel=tf.reshape(tensor_kernel,shape=[self.height,self.num_outputs])

            w_kernel=kernel[:self.kernel_dim,:]

            output=tf.matmul(input_,w_kernel)

            

        

        

        return self.activation(output) #change the activation here

    

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.num_outputs)

##############################################################################

        

  

#layer = TTFC(10,[2,2],[2,5,11],activation="relu")

    

#outt = layer(tf.zeros([1, 10]))