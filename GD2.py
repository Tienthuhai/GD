def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    history = []
    
    for _ in range(num_iterations):
        fx = f(x)
        history.append((x, fx))
        gradient = f_prime(x)
        x = x - learning_rate * gradient
    
    return history

def f(x):
    return x**2 + 6*x + 8

def f_prime(x):
    return 2*x + 6

import matplotlib.pyplot as plt

starting_point = 0
num_iterations = 100
learning_rates = [0.001, 0.01, 0.1, 1.0]

plt.figure(figsize=(12, 8))

for lr in learning_rates:
    history = gradient_descent(starting_point, lr, num_iterations)
    x_values, fx_values = zip(*history)
    
    plt.plot(range(len(fx_values)), fx_values, label=f'Learning rate = {lr}')

plt.xlabel('Số bước lặp')
plt.ylabel('f(x)')
plt.title('Sự hội tụ của Gradient Descent với các learning rate khác nhau')
plt.legend()
plt.grid(True)
plt.show()
