import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import pandas as pd
import numpy as np
import pickle

class Model():
    def __init__(self, args, module='train'):
        self.args = args
        self.pos_length = args.seq_length
        training = True
        reuse = False
        
        if module != 'train':
            training = False
        if module == 'valid':
            reuse = True
            args.batch_size = args.batchSizeValid
        if module == 'sample':
            args.batch_size = 1
            args.seq_length = 1
        
        
            
        # choose different rnn cell 
        if args.model == 'rnn':
            cell_fn = rnn.RNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # warp multi layered rnn cell into one cell with dropout
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # input/target data (int32 since input is char-level)
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        
        with tf.variable_scope('rnnlm', reuse=reuse):
            embedding = tf.get_variable("embedding", [args.vocab_size, args.emb_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        # unstack the input to fits in rnn model
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # loop function for rnn_decoder, which take the previous i-th cell's output and generate the (i+1)-th cell's input
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)
        
        with tf.variable_scope('rnnlm', reuse=reuse):
            # softmax output layer, use softmax to classify
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])##
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])##
            # rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
            outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        # output layer
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        # loss is calculate by the log loss and taking the average.
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)

        if training:
            tvars = tf.trainable_variables()
    
            # calculate gradients
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                    args.grad_clip)
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(self.lr)
    
            # apply gradient change to the all the trainable variable.
            self.model_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.model_op = tf.no_op()

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for _ in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
    
    def evaluate(self, vocab, lines, save_path, save_dir):
        chars = list(lines)
        nextPList = []
        probsList = []
        cellList = []
        hiddenList = []
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('! load model:',ckpt.model_checkpoint_path)
            for i in range(len(chars)-1):
                if chars[i]=='^':
                    state = sess.run(self.cell.zero_state(1, tf.float32))

                x = np.zeros((1, 1))
                x[0, 0] = vocab[chars[i]]
                feed = {self.input_data: x, self.initial_state: state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)

                
                #prob of next char
                if chars[i+1]=='$':
                    nextPList.append(0)
                else:
                    nextPList.append( probs[0][vocab[chars[i+1]]] )
                probsList.append(probs[0])
                cellList.append(state[0][0][0])
                hiddenList.append(state[0][1][0])
            
            nextPList.append(0)
            probsList.append(0)
            cellList.append(0)
            hiddenList.append(0)
        
        return chars, nextPList, probsList, cellList, hiddenList    
        
    def sample2(self, vocab, lines, save_path, save_dir):
        chars = list(lines)
        ret = []
        vocab_size = len(vocab)
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('! load model:',ckpt.model_checkpoint_path)

            for i in range(len(chars)-1):
                if chars[i]=='^':
                    state = sess.run(self.cell.zero_state(1, tf.float32))

                x = np.zeros((1, 1))
                x[0, 0] = vocab[chars[i]]
                feed = {self.input_data: x, self.initial_state: state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                #prob of next char
                ret.append( probs[0][vocab[chars[i+1]]] )
            
            ret.append(0)
            print(probs)

        return chars, ret