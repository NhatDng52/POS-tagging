import tensorflow as tf
import numpy as np

class CustomRNN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomRNN, self).__init__()
        
        # Trọng số
        self.W_hh = tf.Variable(tf.random.normal([hidden_dim, hidden_dim]))  # Ẩn -> Ẩn
        self.W_ih = tf.Variable(tf.random.normal([input_dim, hidden_dim]))   # Input -> Ẩn
        self.b_hh = tf.Variable(tf.zeros([hidden_dim]))                      # Bias ẩn
        
        self.W_ho = tf.Variable(tf.random.normal([hidden_dim, output_dim]))  # Ẩn -> Output
        self.b_ho = tf.Variable(tf.zeros([output_dim]))                      # Bias output

    






























# Thử nghiệm mô hình
# batch_size = 2
# seq_length = 5
# input_dim = 3
# hidden_dim = 4
# output_dim = 2

# rnn = CustomRNN(input_dim, hidden_dim, output_dim)
# test_input = tf.random.normal([batch_size, seq_length, input_dim])
# output = rnn(test_input)
# print(output.shape)  # (batch_size, seq_length, output_dim)

# def call(self, inputs, targets, epochs=1000, learning_rate=0.01):
    #     optimizer = tf.keras.optimizers.Adam(learning_rate)
    #     loss_fn = tf.keras.losses.MeanSquaredError()
    #     for epoch in range(epochs):
    #         with tf.GradientTape() as tape:
    #             Y_pred = rnn(inputs)
    #             loss = loss_fn(targets, Y_pred)
            
    #         grads = tape.gradient(loss, rnn.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, rnn.trainable_variables))
            
    #         if epoch % 10 == 0:
    #             print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")