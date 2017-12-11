import tensorflow as tf
import random, os, sys, time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

samples = 50
seq_length = 30
max_value = 100
output_classes = 2
MINI_BATCH_SIZE = 5
LEARNING_RATE = 0.005
EPOCHS = 2

class data_generator:
    def __init__(self):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(samples):
            if random.random() < .5:
                # Generate a linear sequence
                divisor = random.randint(1, max_value)
                seq_start = random.randint (0, max_value)
                s = [[float(i)/divisor] for i in
                     range(seq_start, seq_start + seq_length)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(seq_length)]
                self.data.append(s)
                self.labels.append([0., 1.])
    


class rnn_net:
    def __init__(self, data_x, labels_y):
        """Creates a rnn_net
            
        :param train or test data
        :param labels
        :param tensorflow model file
        """
        self.x= data_x
        self.y= labels_y

    # ========== #
    #   MODEL    #
    # ========== #       

    def __initialise_variables(network_description, mode, cost):
        """Creates a neural network based on the 
           number of hidden_layers specified in the 
           network_description file

        :param filename
        :param mode, either training or testing
        :param regularisation to apply

        :return created NN architecture and variables
        """
        #placeholder initialiser
        x = tf.placeholder( tf.float32 , [ None , seq_length, 1] , name = "x")

        y = tf.placeholder( tf.float32 , [ None , output_classes ] , name = "y")

        seq = tf.placeholder(tf.int32, [MINI_BATCH_SIZE])

        """
        Inputs need to be unstacked as the RNN accepts a list of input Tensors.
        Unstack is needed to unpack seq_length number of tensors which can accept
        MINI_BATCH_SIZE number of inputs and provide hidden_layers number of outputs
        """
        x1 = tf.unstack(x, num= seq_length, axis= 1)
        
        #Dictionary to store weights and biases
        W={}
        b={}        

        file = open(network_description + '.txt','r')
        variables = file.readlines()
        hidden_layers = int(variables[0].split()[0])

        #Weights and biases for the output layer
        W['out'] = tf.get_variable('W_out', [hidden_layers, output_classes])
        
        b['out'] = tf.get_variable('b_out', [output_classes])

        """The LSTM cell
        The number of outputs of the LSTM cell is equal to the number of hidden units
        """
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layers)

        #initial state
        #init = tf.zeros([MINI_BATCH_SIZE, seq_length], dtype= tf.float32)

        #The RNN
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
                            

        """
        outputs from the above equation is a list of values obtained 
        from the output of the graph at each timestep. The list has
        the same shape as the input placeholder. This is stacked into a  
        Tensor. The dimensions are modified as [MINI_BATCH_SIZE, seq_length, hidden_layers]
        to enable easy computation of the final layer.
        """
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        """ 
        To classify a sequence, the input variables must be fed to the RNN one after the other.
        The RNN will classify the sequence after it has seen all inputs.
        It is therefore necessary to only store the output of the last time-step.
        However, TensorFlow doesn't support advanced indexing yet.
        For every sequence, the output at seq_length is used
        """

        #Batch size needed for training and testing
        batch = tf.shape(outputs)[0]
        
        # Setting the indices to the last time-step of each sequence
        index = tf.range(0, batch) * seq_length + (seq_length - 1)
        # Storing only selected outputs
        outputs = tf.gather(tf.reshape(outputs, [-1, hidden_layers]), index)

        # Linear activation
        outputs= tf.add(tf.matmul(outputs, W['out']) , b['out'])
        
        regularization_dic = {"cross": tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(0.0),  W.values()), 
                    "cross-l1": tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(0.01), W.values()),
                    "cross-l2":  tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), W.values()),
                    "test": 0}    
        
        reg = regularization_dic[cost]

        #cost function
        cf = tf.nn.softmax_cross_entropy_with_logits(
            logits = outputs , labels = y) + reg     

        out_layer= tf.nn.softmax(outputs)
        
        #predicts if the output is equal to its expectation 
        correctness_of_prediction = tf.equal(
            tf.argmax(out_layer, 1), tf.argmax(y, 1))

        #accuracy of the NN
        accuracy = tf.reduce_mean(
            tf.cast(correctness_of_prediction, tf.float32), name='accuracy')

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE) 
        train = optimizer.minimize(cf)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        if mode == 'training':
            return x, y, sess, train, accuracy, out_layer
        elif mode == 'test':
            return x, y, out_layer, accuracy


    # ====================== #
    #   Training algorithm   #
    # ====================== #    

    def __train(params, xdata, ydata, model):
        '''Trains the network with the provided data

        :param all essential training parameters
        :param input data and target values
        :param name of model file to save the created tf graph
        '''
       
        x=params['x']
        y=params['y']
        sess=params['sess']
        train=params['train']
        accuracy=params['accuracy']

        start_time=time.time()
    
        for j in range (EPOCHS):
            training_acc=0
            print("EPOCH NUMBER: ", j+1)
            for k in range(0, len(xdata), MINI_BATCH_SIZE):
                current_batch_x_train = xdata[k:k+MINI_BATCH_SIZE]
                current_batch_y_train = ydata[k:k+MINI_BATCH_SIZE]
                _= sess.run(train,
                        {x: current_batch_x_train, y: current_batch_y_train})           


        train_time=time.time() - start_time
        
        saver= tf.train.Saver()
        saver.save(sess , model)
        
        print("Total training time= ", train_time, "seconds")

    def _5_fold_cross_validation(self, network_description, cost, model):
        '''Performs 5-fold cross validation

        :param text file containing the network description
        :param regularisation to apply
        '''
        x, y, sess, train, accuracy, out_layer = rnn_net.__initialise_variables(network_description, 'training', cost)
        params={}
        params['x']=x
        params['y']=y
        params['sess']=sess
        params['train']=train
        params['accuracy']=accuracy
        params['out_layer']=out_layer
        model = os.path.join(os.getcwd(), model)
        subset_size = len(self.x) // 5
        subsets_x = []
        subsets_y = []
        for i in range(0, len(self.x) , subset_size):
            subset = self.x[i:i+subset_size]
            subsets_x.append(subset)
            subset = self.y[i:i+subset_size]
            subsets_y.append(subset)

        for j in range(5):
            train_set_x = []
            train_set_y = []
            test_set_x = []
            test_set_y = []
            for i in range(5):
                if i != j:
                    train_set_x.extend(subsets_x[i])
                    train_set_y.extend(subsets_y[i])
                    
                else:
                    test_set_x=subsets_x[i]
                    test_set_y=subsets_y[i]                    
            rnn_net.__train(params, train_set_x, train_set_y, model)
            acc, matrix= self.test(model, test_set_x, test_set_y, cost, params)
            print(acc)

            print(matrix)

    # ================= #
    #   Test Function   #
    # ================= # 

    def test(self, model_file, xdata=None, ydata=None, cost=None, params=None, network_description=None):
        """Tests either the model stored in the file or the one currently
           being trained by the 5_fold_cross_validator

           :param: tf neural network
           :return: The model's accuracy, confusion matrix
        """
        if params is None:
            x, y, out_layer, acc=rnn_net.__initialise_variables(network_description, 'test', cost)
            xdata=self.x
            ydata=self.y

        else :
            x=params['x']
            y=params['y']
            out_layer=params['out_layer']
            acc=params['accuracy']

        # Get Tensorflow model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess , model_file)        
        print ("Model restored!")
        
        start_time=time.time()

        total_accuracy = sess.run(acc, {x: xdata, y: ydata})

        #the softmax layer's output
        prediction = tf.argmax(out_layer, 1)
        actual = tf.argmax(ydata, 1)

        pred, act = sess.run([prediction, actual],  {x: xdata, y: ydata})
        #print("Prediction ",pred)
        #print("Actual",act)

        confusion_matrix = sess.run(tf.confusion_matrix(
                act, pred), {x: xdata, y: ydata})

        duration = time.time() - start_time

        if params is None:
            print ("Total number of items tested on: ", len(xdata))
            print ("Total Accuracy over testing data: ", total_accuracy)
            print("Testing time: ", duration, " seconds")
            print("Confusion Matrix:\n", confusion_matrix)

        else :
            return  total_accuracy, confusion_matrix    

if __name__ == '__main__' :
    if len(sys.argv) != 2:
        print ("Please supply the correct arguments")
        raise SystemExit(1)

data_train= data_generator()

rnn_model = rnn_net ( getattr(data_train, 'data'), getattr(data_train, 'labels'))

if str(sys.argv[1]) == 'test' :
    rnn_model.test(model_file = 'model', cost='cross', network_description='network_description')
else:
    rnn_model._5_fold_cross_validation('network_description', 'cross', 'model')
