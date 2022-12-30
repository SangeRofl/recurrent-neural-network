import numpy as np
import random


def leaky_ReLU(matrix):
    res = np.array([])
    for i in range(len(matrix)):
        if matrix[i] < 0:
            res = np.append(res, matrix[i] * 0.1)
        else:
            res = np.append(res, matrix[i])
    return res


def derivative_leaky_ReLU(matrix):
    res = np.array([])
    for i in range(len(matrix)):
        if matrix[i] < 0:
            res = np.append(res, 0.1)
        else:
            res = np.append(res, 1)
    return res


def save_weights(W_1_2, W_c_2, W_2_3):
    pass


def read_weights(W_1_2, W_c_2, W_2_3):
    pass


def hidden_S(input_data, context_data, T = np.array([0, 0])):
    res = np.array([])
    for i in range(len(input_data)):
        res = np.append(res, input_data[i] + context_data[i]-T[0, i])
    return res


def study():
    pass

def create_random_weight_array(rows, cols):
    res = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(random.random() * 2 - 1)
        res.append(row)
    return np.array(res)


if __name__ == "__main__":
    # data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # reference = np.array([4, 5, 6])
    # data = np.array([[1, 3, 5], [3, 5, 7], [5, 7, 9]])
    # reference = np.array([7, 9, 11])
    # data = np.array([[1, 2, 4], [2, 4, 8], [4, 8, 16]])
    # reference = np.array([8, 16, 32])
    data = np.array([[1, 1, 2], [1, 2, 3], [2, 3, 5]])
    reference = np.array([3, 5, 8])
    context_data = np.array([0])
    T_2 = np.array([[0, 0]])
    T_3 = 0
    alpha = 0.01
    input_data2 = np.array([2, 3, 4])
    reference1 = 4
    reference2 = 5
    e = 0.0000000000000000000001
    first_layer_count = 3
    hidden_layer_count = 2
    output_layer_count = 1
    context_layer_count = output_layer_count
    W_1_2 = create_random_weight_array(first_layer_count, hidden_layer_count)
    W_2_3 = create_random_weight_array(hidden_layer_count, output_layer_count)
    W_c_2 = create_random_weight_array(context_layer_count, hidden_layer_count)
    while True:
        E = 0
        for i in range(len(data)):
            input_data = np.array(data[i])
            reference_data = reference[i]
            a1 = input_data @ W_1_2  # 1x3 * 3x2 = 1x2
            a2 = context_data @ W_c_2
            s1 = hidden_S(a1, a2, T_2)
            a3 = leaky_ReLU(s1)
            a4 = a3 @ W_2_3
            res = leaky_ReLU(a4 - T_3)
            context_data = np.array(res)
            W_2_3 = W_2_3 - alpha * (res - reference_data) * np.array([a3]).transpose()
            T_3 = T_3 + alpha * (res[0] - reference_data)
            gamma_matrix = (res - reference_data) * W_2_3
            W_1_2 = W_1_2 - alpha * ((gamma_matrix.transpose() * np.array([derivative_leaky_ReLU(s1)])).transpose() @ np.array([input_data])).transpose()
            W_c_2 = W_c_2 - alpha * gamma_matrix.transpose() * np.array([derivative_leaky_ReLU(s1)]) * 0
            T_2 = T_2 + alpha * gamma_matrix.transpose() * np.array([derivative_leaky_ReLU(s1)])
            E_i = (res[0] - reference_data) ** 2
            E += E_i
            print("res = ", res, "; need = ", reference[i])
        print(E)
        if E <= e:
            break



    a1 = np.array([223, 377, 610]) @ W_1_2  # 1x3 * 3x2 = 1x2
    a2 = context_data @ W_c_2
    s1 = hidden_S(a1, a2, T_2)
    a3 = leaky_ReLU(s1)
    a4 = a3 @ W_2_3
    res = leaky_ReLU(a4 - T_3)
    print(round(res[0]))

    # ITERATION 2
    # context_data2 = res1
    #
    # a1 = input_data2 @ W_1_2
    # a2 = context_data2 @ W_c_2
    # s1 = hidden_S(a1, a2, T_2)
    # a3 = leaky_ReLU(s1)
    # a4 = a3 @ W_2_3
    # res2 = leaky_ReLU(a4 - T_3)
    # E2 = 1 / 2 * (res2[0] - reference2) ** 2
    # print(res2)
    # print(E2)



