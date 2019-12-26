import importlib
import json
from os.path import join

from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.contrib.keras import models as km


def convert_keras_model(model_class, loader_method, model_path, model_inputs):
    """
    Convert keras model to something similar to a yatima model.
    :param model_class (str): The name of a class ot function loader to import
    :param loader_method (str): The method within  the class model_class to load. If the loader is directly a function use model_class and leave this to None
    :param model_path (str): The address where the weights are stored
    :param model_inputs (str): The inputs of the model loader to be able to create the model.
    :return: A dictionary with the model, input and output
    """
    point = model_class.rfind('.')
    package = model_class[:point]
    name = model_class[point+1:]
    if loader_method is None:
        module = importlib.import_module(package)
        networkbuilder = getattr(module, name)
    else:
        module = importlib.import_module(package)
        class_model = getattr(module, name)
        networkbuilder = getattr(class_model(), loader_method)

    keras_model = networkbuilder(*model_inputs)
    keras_model.load_weights(model_path)

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

        return [self.grad_cam(list(feed.values())[0], predicted_class, last_layer)]

        weights_last_layer = last_layer.weights[0]
        if len(last_layer.weights) > 1:  # bias exists
            bias_last_layer = last_layer.weights[1]
            bias_selected_class = tf.slice(bias_last_layer, [predicted_class], [1])
        else:
            bias_selected_class = 0

        if self.separate_negative_positive:
            weights_positive = tf.clip_by_value(weights_last_layer, 0, 1e6)
            signals = [tf.einsum("ij,ja-> ia",layers[i].input, weights_positive)[:,predicted_class]]

            weights_negative = -tf.clip_by_value(weights_last_layer, -1e6, 0)
            signals.append(tf.einsum("ij,ja-> ia",layers[i].input, weights_negative)[:,predicted_class])
        else:
            signals = [tf.einsum("ij,ja-> ia",layers[i].input, weights_last_layer)[:,predicted_class]] #[tf.matmul(layers[i].input, weights_selected_class)]
            signals.append(layers[i].output[:,predicted_class])

        losses = [signal + bias_selected_class for signal in signals]

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
            norm_grads = grads #tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
            cam = self.compute_cam(layer_visualise, norm_grads, feed)
            cam3.append(cam)

        return cam3

    def compute_cam(self, layer_visualise, grads, feed):
        """
        Compute the equation of CAM from the paper.
        :param layer_visualise: The visualisation of the layer
        :param grads:
        :param feed:
        """
        output, grads_val = self.sess.run([layer_visualise, grads], feed_dict=feed)
        output = output[0]
        grads_val = grads_val[0]

        if self.guided_relu:
            grads_val = np.maximum(grads_val, 0)

        if self.no_pooling:
            weights = np.squeeze(grads_val)
        else:
            weights = np.mean(grads_val, axis=tuple(range(len(grads_val.shape) - 1)))  # [512]

        # Taking a weighted average
        if self.no_pooling:
            cam = np.sum(weights * output, axis=2)
        else:
            cam = np.dot(output, weights)

        cam = resize(cam, self.img.shape[0:2])
        cam = np.maximum(cam, 0) #ReLU
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max

        return cam

    def grad_cam(self, image, cls, layer_name):
        """GradCAM method for visualizing input saliency."""
        from tensorflow.python.keras import backend as K
        y_c = layer_name.output[0, cls]
        conv_output = self.model['Model'].get_layer(self.layer_name).output
        grads = K.gradients(y_c, conv_output)[0]
        # Normalize if necessary
        # grads = normalize(grads)
        gradient_function = K.function([self.model['Model'].input], [conv_output, grads])

        output, grads_val = gradient_function([image])
        output, grads_val = output[0, ...], grads_val[0, ...]

        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # Process CAM
        cam = resize(cam, self.img.shape[0:2])
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max != 0:
            cam = cam / cam_max
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

    with open(args.model_args_path, 'r') as f:
        values = json.load(f)

    model, sess = convert_keras_model(values['model_class'], values['loader_method'], values['weight_path'], values['model_inputs'])
    print(model['Model'].summary())

    point = values['process'].rfind('.')
    package = values['process'][:point]
    name = values['process'][point + 1:]

    module = importlib.import_module(package)
    process = getattr(module, name)

    img = process(*values['process_inputs']) #process(imread(args.image_path), image_size, mean_pixel, std_pixel)

    visualiser = GradCam(model, sess, layer_name=args.layer_name, no_pooling=args.no_pooling,
                         guided_relu=args.guided_relu)
    visualiser.last_layer = args.last_layer
    visualiser.select_output = args.select_output
    visualiser.save_path = args.save_image
    visualiser.separate_negative_positive = True

    img2 = img.astype(float)
    img2 /= img2.max()
    visualiser.img = img2[0]

    _ = visualiser.run([img])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_args_path", type=str)
    #parser.add_argument('-model_path', type=str)
    #parser.add_argument('-loader_name', type=str, default=None)
    #parser.add_argument('-model_inputs', type=str, nargs='*')
    #parser.add_argument('-weight_path', type=str)
    parser.add_argument("-layer_name", type=str, default="",
                        help="Name of the layer to visualise. By default the last one, use the input layer when --guided-relu to get the results of the paper")
    parser.add_argument("-select_output", type=int, default=None,
                        help="If the output of the model has more than value (a list of results) select the output")
    parser.add_argument("-last_layer", type=str, default="")
    parser.add_argument("--no_pooling", action="store_true", default=False,
                        help="Average the gradient per layer. This is default as explained in the paper")
    parser.add_argument("--guided_relu", action="store_true", default=False,
                        help="Use guided ReLu instead of standard ReLu. Change the layer_name to the input layer to get results from the paper")
    parser.add_argument("-save_image", type=str, default="",
                        help="Save the grad cam images in the same folder where the image is under the same name with negative and positive features")

    args = parser.parse_args(['-model_args_path','/home/isaac/containers/GradCAM-keras/input_parameters.json',
                             '-select_output',0,
                             '-layer_name','conv2d',
                             '-last_layer', 'out'])
    #input_1, conv2d

    main(args)

