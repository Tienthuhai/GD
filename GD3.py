import numpy as np

def f(x):
    return x**2 + 6*x + 8

def f_prime(x):
    return 2*x + 6

def step_decay(initial_lr, decay_rate, decay_steps, current_step):
    return initial_lr * (decay_rate ** (current_step // decay_steps))

def exponential_decay(initial_lr, decay_rate, current_step):
    return initial_lr * np.exp(-decay_rate * current_step)

def gradient_descent_lr_scheduler(starting_point, initial_learning_rate, num_iterations, 
                                  decay_strategy, decay_rate, decay_steps=None):
    x = starting_point
    history = []
    
    for i in range(num_iterations):
        if decay_strategy == 'step':
            lr = step_decay(initial_learning_rate, decay_rate, decay_steps, i)
        elif decay_strategy == 'exponential':
            lr = exponential_decay(initial_learning_rate, decay_rate, i)
        else:
            lr = initial_learning_rate
        
        fx = f(x)
        history.append((x, fx, lr))
        gradient = f_prime(x)
        x = x - lr * gradient
    
    return history

starting_point = 0
initial_learning_rate = 0.1
num_iterations = 100
decay_rate = 0.9
decay_steps = 10

history_normal = gradient_descent_lr_scheduler(starting_point, initial_learning_rate, num_iterations, 'normal', decay_rate)
history_step = gradient_descent_lr_scheduler(starting_point, initial_learning_rate, num_iterations, 'step', decay_rate, decay_steps)
history_exp = gradient_descent_lr_scheduler(starting_point, initial_learning_rate, num_iterations, 'exponential', decay_rate)

# In kết quả cuối cùng
print("Kết quả cuối cùng:")
for name, history in [('Normal', history_normal), ('Step Decay', history_step), ('Exponential Decay', history_exp)]:
    final_x, final_fx, _ = history[-1]
    print(f"{name}: x = {final_x:.6f}, f(x) = {final_fx:.6f}")
