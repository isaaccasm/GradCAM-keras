from os.path import join

from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.contrib.keras import models as km


def convert_keras_model(config, model_path, keras_model=None):
    """
    Convert keras model to something similar to a yatima model.
    :param keras_model:
    :param image:
    :return:
    """
    if not keras_model:
        define = yd([])
        networkbuilder = getattr(define, config['network'])

        num_classes = config['num_classes']
        image_size = config['dims']
        net = networkbuilder(image_size, num_classes)

        keras_model = net['model']
        keras_model.load_weights(join(model_path, 'model'))

        keras_model.compile('adam', loss='categorical_crossentropy')

    model = {}

    model['Input'] = keras_model.input
    model['Model'] = km.Model(inputs=keras_model.input, outputs=keras_model.output)
    model['Output'] = keras_model.output

    return model, None


class GradCam(object):
    """
    Implementation of the grad-cam in tensorflow
    paper: Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization
    URL: https://arxiv.org/pdf/1610.02391v1.pdf
    """

    def __init__(self, model, sess=None, layer_name='', separate_negative_positive=True, no_pooling=False,
                 guided_relu=False,
                 select_output=None):
        self.model = model
        self.sess = None
        self.eager = False
        self.img = None

        self.layer_name = layer_name
        self.separate_negative_positive = separate_negative_positive
        self.no_pooling = no_pooling
        self.save_path = None

        self.select_output = select_output
        self.last_layer = ''
        self.guided_relu = guided_relu
        self.keras = False

        self.not_closed = False

        if self.sess is None:
            self.sess = tf.keras.backend.get_session()
            self.keras = True

    def grad_cam_keras(self, feed, predicted_class):
        """
        This computes the standard grad_cam or a new version where the positive and negative features can be visualised.
        Grad cam algorithm:  https://arxiv.org/pdf/1610.02391.pdf
        The new version just separate the last layer with the features that add positive contribution to the probabilities
        and those that add negative values.
        :param img: A numpy array with the image to study
        :param predicted_class: The predicted class from the model to study
        :param separate_negative_positive: When True the result is two maps one for the part of the image influencing
                                        positively and the others for the parts influencing negatively
        :return: A list with 1 or 2 numpy arrays, representing the the grad cam image/s.
        """
        print("Setting gradients to 1 for target class and rest to 0")
        # Conv layer tensor
        layers = self.model['Model'].layers
        last_layer = self.model['Model'].layers[-1]
        if self.last_layer:
            # last_layer = self.model['Model'].get_layer(self.last_layer)
            i = -1
            while last_layer.name != self.last_layer:
                i = i - 1
                last_layer = self.model['Model'].layers[i]
        else:
            i = -1
            while len(last_layer.weights) == 0:
                i = i - 1
                last_layer = self.model['Model'].layers[i]

        penultime = i - 1

        weights_last_layer = last_layer.weights[0]
        weights_selected_class = tf.slice(weights_last_layer, [0, predicted_class], [weights_last_layer.shape[0], 1])

        if len(last_layer.weights) > 1:  # bias exists
            bias_last_layer = last_layer.weights[1]
            bias_selected_class = tf.slice(bias_last_layer, [predicted_class], [1])
        else:
            bias_selected_class = 0

        if self.separate_negative_positive:
            weights_positive = tf.clip_by_value(weights_selected_class, 0, 1e6)
            signals = [tf.multiply(weights_positive, layers[penultime].output)]

            weights_negative = -tf.clip_by_value(weights_selected_class, -1e6, 0)
            signals.append(tf.multiply(weights_negative, layers[penultime].output))
        else:
            signals = [tf.multiply(weights_selected_class, layers[penultime].output)]

        losses = [tf.reduce_sum(signal) + bias_selected_class for signal in signals]

        # Now it is time for the gradients. Notice, that the gradients of the output for a single layer before softmax
        # correspond to the weights of the layer.
        # in_last_layer = True
        if len(self.layer_name) == 0:
            self.layer_name = 'conv'

        for i in range(len(layers) - 1, -1, -1):
            if layers[i].name.lower().find(self.layer_name) > -1:
                layer_visualise = layers[i].output
                break

        cam3 = []
        for loss in losses:
            grads = tf.gradients(loss, layer_visualise)[0]
            # Normalizing the gradients
            norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

            cam = self.compute_cam(layer_visualise, norm_grads, feed)

            cam3.append(cam)

        return cam3

    def compute_cam(self, layer_visualise, grads, feed):
        output, grads_val = self.sess.run([layer_visualise, grads], feed_dict=feed)
        output = output[0]
        grads_val = grads_val[0]

        if self.no_pooling:
            weights = np.squeeze(grads_val)
            weights = weights / np.min(weights, axis=2)[:, :, np.newaxis]
            weights /= np.sqrt(np.mean(weights ** 2, axis=2))[:, :, np.newaxis]
        else:
            grads_val /= np.min(grads_val)  # TO AVOID that the square of grad_vals is 0.
            grads_val /= np.sqrt(np.mean(grads_val ** 2))
            weights = np.mean(grads_val, axis=tuple(range(len(grads_val.shape) - 1)))  # [512]
        cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

        # If there is nan means that the gradients were 0 and therefore the weights should 0
        weights[np.isnan(weights)] = 0

        # Passing through ReLU
        output = np.maximum(output, 0)

        if self.guided_relu:
            weights = np.maximum(weights, 0)

        # Taking a weighted average
        if self.no_pooling:
            for i in range(weights.shape[2]):
                cam += weights[:, :, i] * output[:, :, i]
        else:
            for i, w in enumerate(weights):
                cam += w * output[:, :, i]

        if np.max(cam) > 0:
            cam = cam / (np.max(cam) + 1e-12)

        return cam

    def visualise_cams(self, cams):
        """
        Show the cams with the real image
        :param cams: A list with the cams
        :return: None
        """
        names = ['Global']
        pooling = 'pooling'
        if len(cams) > 1:
            names = ['Positive', 'Negative']
        if self.no_pooling:
            pooling = 'no pooling'
        for i in range(len(cams) + 1):
            if i > 0:
                cam = resize(cams[i - 1], self.img.shape[0:2])

            plt.figure()
            plt.imshow(np.squeeze(self.img))
            if i > 0:
                plt.imshow(cam, cmap='jet', alpha=0.5)
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False
            )
            if i > 0:
                plt.title(
                    '{} features with {} --- Predicted class: {}'.format(names[i - 1], pooling, self.predicted_class))
                if self.save_path is not None and len(self.save_path) > 0:
                    plt.savefig(self.save_path + '_' + names[i - 1] + '-features.jpg')
            else:
                plt.title('Original image')
        plt.show()

    def run(self, inputs, class_position=None):
        """
        Run the process
        :param inputs: A list with all the inputs of the network. They must have the same format and order as it was used for training
        :param layer_name: A string with the name of the layer tovisualise. By default, the last CNN will be used.
        :param class_position: The position of the class to study. By default, None, meaning the class that produces the highest response.
        :return: The grad cam/s imges
        """

        if self.keras:
            # init_op = tf.global_variables_initializer()
            # self.sess = tf.Session()
            # self.sess.run(init_op)
            with self.sess.as_default():
                input_layers = [self.model[layer] for layer in self.model if layer.lower().find('input') > -1]
                feed = {layer: data for layer, data in zip(input_layers, inputs)}
                if self.select_output is not None:
                    prob = self.sess.run(self.model['Output'][self.select_output], feed_dict=feed)[0]
                else:
                    prob = self.sess.run(self.model['Output'], feed_dict=feed)[0]

            grad_cam = self.grad_cam_keras
        else:

            with self.sess.as_default():
                # operations = self.sess.graph.get_operations()

                feed = {layer: data for layer, data in zip(self.model['Inputs'], inputs)}

                prob = np.squeeze(self.sess.run(self.model['y_conv'], feed_dict=feed))
                prob2 = np.exp(prob)
                if np.all(np.isfinite(prob2)):
                    prob = prob2 / np.sum(prob2)
                else:
                    prob2 = np.zeros(prob.shape[0])
                    prob2[np.argmax(prob)] = 1
                    prob = prob2
                grad_cam = self.grad_cam_tf

        preds = (np.argsort(prob)[::-1])[0:5]

        # Target class
        self.predicted_class = preds[0]
        if class_position:
            self.predicted_class = class_position

        if not isinstance(self.predicted_class, (int, np.int_, np.ushort)):
            raise ValueError('Predicted class value must be an integer')

        outputs = grad_cam(feed, self.predicted_class)

        if self.img is not None:
            self.visualise_cams(outputs)

        if not self.not_closed:
            self.sess.close()
        return outputs


def main(args):
    config = ytt.read_config(join(args.model_path, 'deploy.cfg'))
    if config.get('global_params', None) != None:
        configs = []
        for key, value in config.items():
            if 'global_param' not in key:
                value['name'] = key
                configs.append(value)
        config = configs[0]
    else:
        config['name'] = ''

    mean_pixel = config['mean_pixel']
    std_pixel = config['std_pixel']
    image_size = config['dims']

    if args.not_yatima_model:
        model, sess = convert_keras_model(config, join(args.model_path, config['name']))
    else:
        model, sess = read_yatima_model(config, join(args.model_path, config['name']))

    img = process(imread(args.image_path), image_size, mean_pixel, std_pixel)

    name_save = args.image_path[:args.image_path.rfind('.')]

    visualiser = GradCam(model, sess, layer_name=args.layer_name, no_pooling=args.no_pooling,
                         guided_relu=args.guided_relu)
    visualiser.last_layer = args.last_layer
    visualiser.select_output = args.select_output

    if args.save_image:
        visualiser.save_path = name_save

    img2 = img.astype(float)
    img2 /= img2.max()
    visualiser.img = img2[0]

    _ = visualiser.run([img])




