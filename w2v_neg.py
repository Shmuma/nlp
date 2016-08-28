import tensorflow as tf
import math

if __name__ == "__main__":
    dict_size = 10
    vec_size = 100
    batch_size = 50
    num_sampled = 20

    embeddings = tf.Variable(tf.random_uniform([dict_size, vec_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))
    nce_biases = tf.Variable(tf.zeros([dict_size]))

    # Placeholders for inputs
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Compute the NCE loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, dict_size))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)


