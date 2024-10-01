import numpy as np

def f(x):
    return x**2 + 6*x + 8

def f_prime(x):
    return 2*x + 6

def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    results = []
    
    for _ in range(num_iterations):
        fx = f(x)
        results.append((x, fx))
        
        gradient = f_prime(x)
        x = x - learning_rate * gradient
    
    results.append((x, f(x)))
    
    return results


starting_point = 0
learning_rate = 0.1
num_iterations = 20

results = gradient_descent(starting_point, learning_rate, num_iterations)

print("Iteration\t x\t\t f(x)")
print("-" * 40)
for i, (x, fx) in enumerate(results):
    print(f"{i}\t\t {x:.6f}\t {fx:.6f}")

final_x, final_fx = results[-1]
print(f"\nGiá trị x tại cực tiểu: {final_x:.6f}")
print(f"Giá trị cực tiểu của hàm f(x): {final_fx:.6f}")
