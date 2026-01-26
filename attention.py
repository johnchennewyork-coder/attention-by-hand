

import numpy  as np
# also need to import softmax from scipy

print("numpy imported")

def numpy_softmax(x, axis=-1):
    # Subtracting the max for numerical stability (the log-sum-exp trick)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def attention(Q,K,V, mask=None):
    '''
    The attention equation is attention (Q,K,V) = softmax(QK^T)/sqrt(d_key) * V   

    We also have multihead attention.

    Let's look at the matrices that define the attention equation. 
    Q \in R^{BS X SEQ_LEN X D_MODEL}
    K \in R^{BS X SEQ_LEN X D_MODEL}
    V \in R^{BS X SEQ_LEN X D_MODEL}


    :param Q: Description
    :param K: Description
    :param V: Description
    :param mask: Description
    '''

    KT =  K.transpose(0, -1,-2)
    attention_weights = numpy_softmax(np.matmul(Q, KT)/np.sqrt(K.shape[-1]), axis=-1) 
    output = np.matmul(attention_weights, V)
    return output, attention_weights

def generate_dummy_data(batch_size, seq_len, d_model):
    Q = np.random.rand(batch_size, seq_len, d_model)
    K = np.random.rand(batch_size, seq_len, d_model)
    V = np.random.rand(batch_size, seq_len, d_model)
    return Q, K, V

def main():
    Q, K, V = generate_dummy_data(2, 4, 8) # 2 examples of 4 words of 8 dimensions each
    # attention(Q, K, V)
    print(attention(Q, K, V)[0].shape, attention(Q, K, V)[1].shape)  # Should print (2, 4, 8) and (2, 4, 4)

if __name__ == "__main__":
    main()

