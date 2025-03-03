#import torch
#print(torch.cuda.is_available())

import tensorflow as tf
# Lista GPUs disponíveis
print("GPUs disponíveis: ", tf.config.list_physical_devices('GPU'))

# Lista GPUs disponíveis
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Define limite de memória para evitar OOM
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # Ajuste conforme necessário
        )
    except RuntimeError as e:
        print(e)

# Executa uma operação para testar o uso da GPU
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)

print("Dispositivo onde a operação foi executada: ", c.device)
print("Operação concluída!")

