import tensorflow as tf
import keras.backend as K


def summarize_gradients(model):
    """
    Add summaries of gradients
    :param model: compiled keras model
    """
    gradients = model.optimizer.get_gradients(model.total_loss, model._collected_trainable_weights)
    for var, grad in zip(model._collected_trainable_weights, gradients):
        n = var.name.split(':', maxsplit=1)[0]
        tf.summary.scalar("gradrms_" + n, K.sqrt(K.mean(K.square(grad))))


def summary_value(name, value, writer, step_no, target_type=float):
    """
    Add given actual value to summary writer
    :param name: name of value to add
    :param value: scalar value
    :param writer: SummaryWriter instance
    :param step_no: global step index
    """
    summ = tf.Summary()
    summ_value = summ.value.add()
    summ_value.simple_value = target_type(value)
    summ_value.tag = name
    writer.add_summary(summ, global_step=step_no)
