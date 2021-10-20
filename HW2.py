import matplotlib.pyplot as plt
import numpy as np

import checker
import utils

# Proszę nie zmieniac, to nam zapewni, ze bedziemy pracowac na tych samych datasetach
np.random.seed(60)

# Pierwszy zbiór wygenerujemy, samplując z rozkładu jednostajnego
train_set_1d = np.random.uniform(-5, 5, size=(15, 1))

# Drugi zbiór wygenerujemy, samplując z rozkładu normalnego
train_set_2d = np.random.normal(-1, 3, size=(13, 2))


# Trzeci zbiór wygenerujemy, samplując z rozkładu wykładniczego
train_set_10d = np.random.exponential(2, size=(31, 10))
print(train_set_10d[:5, 0])

train_sets = [train_set_1d, train_set_2d, train_set_10d]

print("Parę punktów z datasetu jednowymiarowego:\n", train_set_1d[:5])
print("Parę punktów z datsetu dwuwymiarowego:\n", train_set_2d[:5])
print("Parę punktów z datsetu dziesięciowymiarowego:\n", train_set_10d[:3])

shapes = [dataset.shape for dataset in train_sets]
print("Rozmiary naszych datasetów:", *shapes)
# Pierwszy wymiar reprezentuje liczbę przykładów w datasecie
# Drugi wymiar to wymiar pojedynczego przykładu z datasetu

# Sprawdźmy czy datasety wylosowały się z poprawnego seedu.
similar_vals_1d = np.all(np.isclose(
    train_set_1d[:5, 0],
    np.array([-1.9912666995338126, -3.13054183658122, -1.7681732303178057, 1.6574957028830903, 0.6697080096921333])
))

similar_vals_2d = np.all(np.isclose(
    train_set_2d[:5, 0],
    np.array([-6.53671154, -0.96583933, 3.14309184, -0.54717522, -2.49814889])
))

similar_vals_10d = np.all(np.isclose(
    train_set_10d[:5, 0],
    np.array([1.92650794, 2.29209159, 0.61110025, 3.247719, 2.8237559])
))
assert similar_vals_1d and similar_vals_2d and similar_vals_10d, \
    "Wylosowane wartości są złe! Zgłoś od razu problem prowadzącemu!"

print("10d shape: ", train_set_10d[30,9])


########################################ZAD1############################################
def mean_error(X, v):
    vector = X-v.transpose()
    num_rows = np.shape(X)[0]
    row_norms = np.linalg.norm(vector, axis=1)
    sum = np.sum(row_norms)
    div_sum = sum / num_rows
    return div_sum

def mean_squared_error(X, v):
    vector = X-v.transpose()
    num_rows = np.shape(X)[0]
    row_norms = np.linalg.norm(vector, axis=1)
    row_norms_squared = np.square(row_norms)
    sum = np.sum(row_norms_squared)
    div_sum = sum / num_rows
    return div_sum

def max_error(X, v):
    vector = X-v.transpose()
    num_rows = np.shape(X)[0]
    row_norms = np.linalg.norm(vector, axis=1)
    max = np.max(row_norms)
    return max

checker.check_1_1(mean_error, mean_squared_error, max_error, train_sets)

#fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#utils.plot_1d_set(train_set_1d, ax, [mean_error, mean_squared_error, max_error])
#fig.legend()
#plt.show(fig)
#plt.close(fig)
#
#fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#utils.plot_1d_set(train_set_1d, axes[0], [mean_error], show_title=True)
#utils.plot_1d_set(train_set_1d, axes[1], [mean_squared_error], show_title=True)
#utils.plot_1d_set(train_set_1d, axes[2], [max_error], show_title=True)
#plt.show(fig)
#plt.close(fig)
#
#
#utils.plot_2d_loss_fn(mean_error, "Mean Error", train_set_2d)
#utils.plot_2d_loss_fn(mean_squared_error, "Mean Square Error", train_set_2d)
#utils.plot_2d_loss_fn(max_error, "Max Error", train_set_2d)


###################################ZAD2###################################

def minimize_mse(X):
    # średnia z danych
    return np.mean(X)

def minimize_me(X):
    # mediana
    return np.median(X)

def minimize_max(X):
    #w = (min X + max X)/2
    return (np.max(X)+np.min(X))/2

# Testy do zadania
checker.check_1_2(minimize_me, minimize_mse, minimize_max, train_set_1d)


##################################ZAD3################################
def me_grad(X, v):
    matrix = v.transpose()-X
    N = np.shape(X)[0]
    column = np.sqrt(np.sum(np.square(matrix),axis=1))
    modified_column = 1/column
    modified_matrix = matrix*modified_column[:, np.newaxis]
    grad = np.sum(modified_matrix,axis=0)/N
    return grad

def mse_grad(X,v):
    matrix = v.transpose()-X
    N = N = np.shape(X)[0]
    grad = np.sum(matrix,axis=0)*2/N
    return grad

def max_grad(X,v):
    matrix = X-v.transpose()
    column = np.sqrt(np.sum(np.square(matrix),axis=1))
    max_index_col = np.argmax(column)
    denominator = np.sqrt(np.sum(np.square(matrix[max_index_col,])))
    grad = -matrix[max_index_col,]/denominator
    return grad




print(me_grad(train_set_2d,np.array([3])))
print(mse_grad(train_set_2d,np.array([3])))
print(max_grad(train_set_2d,np.array([3])))

checker.check_1_3(me_grad, mse_grad, max_grad, train_sets)

####################################ZAD4####################################################
def gradient_descent(grad_fn, dataset, learning_rate=0.2, num_steps=100):
    """
    grad_fn: funkcja z poprzedniego zadania - przyjmuje dataset 
    dataset: zbiór treningowy na którym trenujemy
    learning_rate: prędkość uczenia, określa jak długi krok gradientu mamy robić
    num_steps: liczba kroków metody.
    """
    current_v = np.random.normal(4, size=(dataset.shape[1]))
    all_v = [current_v]
        
    for step_idx in range(num_steps):
        grad = grad_fn(dataset,all_v[-1])  # liczenie gradientu
        current_v = all_v[-1] - learning_rate*grad  # krok metody gradientu
        
        all_v += [current_v]
        if np.linalg.norm(all_v[-1] - all_v[-2]) < 1e-3:
            break
        
    final_grad = grad
    final_v = current_v
    all_v = np.array(all_v)
    return final_v, final_grad, all_v


fig, ax = plt.subplots(1, 1)
utils.plot_gradient_steps_1d(
    ax, train_set_1d,
    gradient_descent, mse_grad, mean_squared_error,
    learning_rate=0.2, num_steps=10000)
ax.set_title("1d run, lr=0.2")

fig, ax = plt.subplots(1, 1)
utils.plot_gradient_steps_1d(
    ax, train_set_1d,
    gradient_descent, mse_grad, mean_squared_error,
    learning_rate=0.3, num_steps=10000)
ax.set_title("1d run, lr=0.3")

fig, ax = plt.subplots(1, 1)
utils.plot_gradient_steps_1d(
    ax, train_set_1d,
    gradient_descent, mse_grad, mean_squared_error,
    learning_rate=0.4, num_steps=10000)
ax.set_title("1d run, lr=0.4")
fig, ax = plt.subplots(1, 1)
utils.plot_gradient_steps_1d(
    ax, train_set_1d,
    gradient_descent, mse_grad, mean_squared_error,
    learning_rate=0.5, num_steps=10000)
ax.set_title("1d run, lr=0.5")

fig, ax = plt.subplots(1, 1)
utils.plot_gradient_steps_1d(
    ax, train_set_1d,
    gradient_descent, mse_grad, mean_squared_error,
    learning_rate=0.9, num_steps=10000)
ax.set_title("1d run, lr=0.9")

fig, ax = plt.subplots(1, 1)
utils.plot_gradient_steps_1d(
    ax, train_set_1d,
    gradient_descent, mse_grad, mean_squared_error,
    learning_rate=1.0, num_steps=10000)
ax.set_title("1d run, lr=1.0")

fig, ax = plt.subplots(1, 1)
utils.plot_gradient_steps_1d(
    ax, train_set_1d,
    gradient_descent, mse_grad, mean_squared_error,
    learning_rate=1.1, num_steps=10000)
ax.set_title("1d run, lr=1.1")

plt.show(fig)
plt.close(fig)