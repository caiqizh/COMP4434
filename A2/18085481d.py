import math
import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def hx(theta_t_x):
    return 1/(1 + math.exp(-theta_t_x))


def loss(x, y, theta):
    sum = 0
    for i in range(len(x)):
        sum += y[i] * np.log(hx(np.dot(x[i], theta))) + (1 - y[i]) * np.log(1 - hx(np.dot(x[i], theta)))

    return -(1 / m) * sum



data = [[1, -0.12, 0.3, -0.01], [1, 0.2, -0.03, -0.35], [1, -0.37, 0.25, 0.07], [1, -0.1, 0.14, -0.52]]
label = [1, 0, 0, 1]

theta = [-0.09, 0, -0.19, -0.21]
r = 0.2
m = len(data)

cost_list = []

for iteration in range(50000):

    # make a prediction
    pred_label = []
    for item in data:
        theta_t_x =np.dot(item, theta)
        pred_i = hx(theta_t_x)

        if pred_i >= 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)



    # calculate loss

    cost = loss(data, label, theta)
    cost_list.append(cost)
    print(iteration, "iteration, the cost is", cost, "the theta is", theta)
    # update the theta using gradient descent algorithm

    new_theta = []

    for i in range(m):

        sum = 0

        for item in data:
            theta_t_x = np.dot(item, theta)

            pred_i = hx(theta_t_x)

            sum += (pred_i - label[data.index(item)]) * item[i]

        new_theta.append(theta[i] - r * (1 / m) * sum)
        # print(round(theta[i], 3 ), "-", r, "* (1 / 4) *", round(sum, 3), "=", round(theta[i] - r * (1 / m) * sum, 3))

    theta = new_theta

y = cost_list
x = range(len(y))

plt.plot(y)
plt.show()
