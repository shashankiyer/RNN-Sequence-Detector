import tensorflow as tf
import random, os, sys


samples = 50


class data_generator:
    def __init__(self):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])

class rnn_net:
    def __init__(self, data_folder):
        """Creates a conv_net
            
        :param data folder to train with
        :param train or test data
        :param tensorflow model file
        """
        if os.path.isdir(data_folder) is not True:
            print ("NO DATA")
            exit(1)
            
        self.import_data(data_folder)
        self.model_file = os.path.join(os.getcwd(), model_file)

    def __image_reshape(temp_image):
        """Flattens the image into a 
           1D list

        :param The image to flatten
        :return 1D list
        """

        reshaped = np.array(temp_image.getdata()).reshape(temp_image.size[0], temp_image.size[1], 1)
        reshaped = reshaped.tolist()

        return reshaped


    def import_data(self, data_folder):
        """Extracts image data and stores it in a list
           label's the image

        :param data folder to train with
        """
        self.x=[]
        self.y=[]

        #Expression to identify positive samples
        example = re.compile("[0-9]+")

        print("Reading input data")

        data_folder = os.path.join( os.getcwd() , data_folder)
        for images in os.listdir(data_folder):
            label = int (re.match(prog, images))
            images = os.path.join( data_folder , images)
            temp_image = Image.open(images)
            self.x.append(conv_net.__image_reshape(temp_image))

        #randomises data
        for i in range(1, len(self.x)-1):
            j = random.randint(i+1, len(self.x)-1)
            self.swap_t(i, j)
        
        print("Data extraction complete")

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
        x = tf.placeholder( tf.float32 , [ None , seq_length , 1] , name = "x")
        
        y = tf.placeholder( tf.float32 , [ None , output_classes ] , name = "y")

        #Dictionary to store weights and biases
        W={}
        b={}        

        file = open(network_description + '.txt','r')
        variables = file.readlines()
        hidden_layers = variables[0].split()[0]

        #Weights and biases for the output layer
        W['out'] = tf.get_variable('W_out', [hidden_layers, output_classes])
        
        b['out'] = tf.get_variable('b_out', [output_classes])

        #The LSTM cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layers)

        #initial state
        init = tf.zeros([None, lstm_cell.state_size], dtype= tf.float32)

        #The RNN
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, initial_state= init, dtype=tf.float32,
                                sequence_length=seq_length)

        #reshape output
        outputs= tf.reshape(outputs, [-1, hidden_layers])

        #Using all the outputs to cumulatively calculate a softmax
        outputs= tf.add( tf.matmul( outputs, W['out']), b['out'])
        
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

    def _5_fold_cross_validation(self, network_description, cost):
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
            conv_net.__train(params, train_set_x, train_set_y, self.model_file)
            acc, matrix= self.test(self.model_file, test_set_x, test_set_y, cost, params)
            print(acc)

            print(matrix)

