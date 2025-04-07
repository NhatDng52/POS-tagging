import tensorflow as tf
import numpy as np

class CustomRNN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomRNN, self).__init__()
        
        # Trọng số
        self.W_hh = self.add_weight(shape=(hidden_dim, hidden_dim), initializer='random_normal', trainable=True)
        """h,h"""
        self.W_ih = self.add_weight(shape=(input_dim, hidden_dim), initializer='random_normal', trainable=True)   # Input -> Ẩn
        """i,h"""
        self.b_hh = self.add_weight(shape=(hidden_dim,), initializer='zeros', trainable=True)                      # Bias ẩn
        """h (can broadcast)"""
        self.W_ho = self.add_weight(shape=(hidden_dim, output_dim), initializer='random_normal', trainable=True)  # Ẩn -> Output
        """h,o"""
        self.b_ho = self.add_weight(shape=(output_dim,), initializer='zeros', trainable=True)                      # Bias output
        """o (can broadcast)"""

    def call(self, inputs):
        """ inputs : list(sequence) trong đó seq là và vector của 1 mẫu 
            labels : list(label) trong đó label là và vector của 1 mẫu """
        # print(inputs.shape)
        batch_size, seq_length, input_dim = inputs.shape 
        """ b,s,x"""
        hidden_dim = self.W_hh.shape[0]
        """h"""
        
        # Khởi tạo hidden state
        h_t = tf.zeros([batch_size, hidden_dim])
        """b,h"""
        outputs = []
        for t in range(seq_length):
            inputs = tf.cast(inputs, tf.float32)
            x_t = inputs[:, t, :]  # Lấy từng bước thời gian
            "b,x"
            
            # Công thức RNN
            h_t = tf.tanh(tf.matmul(h_t, self.W_hh) + tf.matmul(x_t, self.W_ih) + self.b_hh)
            """
            b,h = tanh((b,h * h,h) + (b,x * i,h) + h) 
                = tanh(b,h + b,h + h)    if x == i
                = b,h  (after broadcast)
            """
            y_t = tf.matmul(h_t, self.W_ho) + self.b_ho  # Tính đầu ra
            """
            b,o = (b,h * h,o) + o 
            """
            outputs.append(y_t)
            # print(tf.stack(outputs, axis=1).shape)
        return tf.stack(outputs, axis=1)  # Trả về chuỗi đầu ra theo thời gian
    def train_model(self, X_train, Y_train, epochs=200, learning_rate=0.1, batch_size=32):
        """
        Huấn luyện mô hình với dữ liệu X_train và Y_train theo batch.
        - X_train: Tensor có shape (num_samples, seq_length, input_dim)
        - Y_train: Tensor có shape (num_samples, seq_length, num_classes) (one-hot)
        Sử dụng loss function CategoricalCrossentropy (from_logits=True) vì output chưa qua softmax.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in range(num_batches):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, num_samples)
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]
                
                with tf.GradientTape() as tape:
                    Y_pred = self(X_batch)  # (batch, seq_length, output_dim)
                    loss = loss_fn(Y_batch, Y_pred)
                
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                
                # Cộng dồn loss theo số mẫu trong batch
                epoch_loss += loss.numpy() * (end - start)
            
            epoch_loss /= num_samples
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")