import tensorflow as tf

# Função de perda personalizada com pos_weight
def custom_binary_crossentropy(pos_weight):
    def loss(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=pos_weight)
    return loss
