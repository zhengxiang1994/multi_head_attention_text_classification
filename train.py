from multihead import *
import data_helper
import numpy as np
from tensorflow.contrib import learn

# Data loading params
tf.flags.DEFINE_string('data_file_path', './data/rt-polarity.csv', 'Data source')
tf.flags.DEFINE_string('feature_name', 'comment_text', 'The name of feature column')
tf.flags.DEFINE_string('label_name', 'label', 'The name of label column')
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

FLAGS = tf.flags.FLAGS


def tokenizer(docs):
    for doc in docs:
        yield doc.split(' ')


def pre_process():
    # load data
    x_text, y = data_helper.load_data_and_labels(FLAGS.data_file_path, FLAGS.feature_name, FLAGS.label_name)
    # Build vocabulary and cut or extend sentence to fixed length
    max_document_length = max([len(x) for x in tokenizer(x_text)])
    print('max document length: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=tokenizer)
    # replace the word using the index of word in vocabulary
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


class AttentionClassifier(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph...")
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        # multi-head attention
        ma = multihead_attention(queries=batch_embedded, keys=batch_embedded, dropout_rate=1-self.keep_prob)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.layers.dense(outputs, units=self.n_class)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        self.prediction = tf.argmax(tf.nn.softmax(logits), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


if __name__ == '__main__':
    # load data
    x_train, y_train, vocab_processor, x_dev, y_dev = pre_process()
    vocab_size = len(vocab_processor.vocabulary_)

    config = {
        "max_len": x_train.shape[1],
        "hidden_size": 32,
        "vocab_size": vocab_size, 
        "embedding_size": 128,
        "n_class": y_train.shape[1],
        "learning_rate": 5e-4,
        "batch_size": 32,
        "train_epoch": 5
    }

    classifier = AttentionClassifier(config)
    classifier.build_graph()

    # accuracy
    correct_prediction = tf.equal(classifier.prediction, tf.argmax(classifier.label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # train
        for epoch_i in range(config['train_epoch']):
            for batch_i, (x_batch, y_batch) in enumerate(
                    data_helper.get_batches(x_train, y_train, config['batch_size'])):
                _, acc, loss = sess.run([classifier.train_op, accuracy, classifier.loss],
                                        feed_dict={classifier.x: x_batch, classifier.label: y_batch,
                                                   classifier.keep_prob: 0.8})
                if batch_i % 10 == 0:
                    print('Epoch {}/{}, Batch {}/{}, loss: {}, accuracy: {}'.format(epoch_i, config['train_epoch'],
                                                                                    batch_i,
                                                                                    len(x_train) // config[
                                                                                        'batch_size'],
                                                                                    loss, acc))
        # valid step
        print('valid accuracy: {}'.format(
            sess.run(accuracy, feed_dict={classifier.x: x_dev, classifier.label: y_dev, classifier.keep_prob: 1.})))
