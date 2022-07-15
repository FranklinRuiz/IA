# ------------------------
# Cargando modelo de disco
# ------------------------
import tensorflow as tf
from keras.models import load_model

def cargarModelo():
    MODEL_PATH = r"D:\utp\IA\proyecto\output\pneumonia_model_full.h5"
    # Cargar la RNA desde disco
    loaded_model = load_model(MODEL_PATH)
    print("Modelo cargado de disco << ", loaded_model)

    #tf.compat.v1.enable_eager_execution()
    graph = tf.compat.v1.get_default_graph()
    return loaded_model, graph