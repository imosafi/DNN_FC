import numpy as np

class GradientsCalculator(object):

    @staticmethod
    def calc_gradients(x, y, y_pred, params):
        gradients_values = []
        y_label = np.zeros(params[-1][1].shape)
        y_label[y] = 1

        # y_pred = classifier_output(x, params)

        z_l, a_l = GradientsCalculator.__calc_params_for_gradients(x, params)
        g_last_b = y_pred - y_label

        gradients_values.insert(0, (GradientsCalculator.__calc_weight_matrix_grad(a_l[-2], g_last_b), g_last_b))

        i = len(params) - 2
        while i > -1:
            gb = GradientsCalculator.__calc_weight_vector_grad(gradients_values[0][1], params[i + 1][0], a_l[i + 1])
            gw = GradientsCalculator.__calc_weight_matrix_grad(z_l[i], gb)
            gradients_values.insert(0, [gw, gb])
            i -= 1
        return gradients_values

    @staticmethod
    def __calc_weight_matrix_grad(x, gb):
        return np.array([x]).transpose().dot([gb])

    @staticmethod
    def __calc_weight_vector_grad(grad, mat, a):
        return grad.dot(mat.transpose()) * (1 - np.power(a, 2))

    @staticmethod
    def __calc_params_for_gradients(x, params):
        z_l = [x]
        a_l = [x]

        i = 0
        while i < len(params):
            z_l.append(np.array(z_l[-1]).dot(params[i][0]) + params[i][1])
            a_l.append(np.tanh(z_l[-1]))
            i += 1
        return z_l, a_l
