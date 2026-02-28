import sys
import numpy as np
'''
Exercise 4
write algorithm for 
bisection
Newton's 
Secant
Hybrid: starts with bisection for early iterations and switches to newton's.

INPUT
The program should take as input a file which contains description of a polynomial in the following format:
n
a(n) a(n-1) a(n-2) ... a(2) a(1) b

ex: 3x^3 + 5x^2 - 7
3
3  5  0 -7

OUTPUT
The file should have extension .pol, for example, fun1.pol would be a suitable name for a file. The program should use bisection method by default and should place the solution in a file with the same name as the input, but with extension .sol (such as fun1.sol), with format:
root  iterations outcome

ADDITIONAL PROGRAM REQUIREMENTS
The program should use bisection as default and operate as follows:

> polRoot [-newt, -sec, -hybrid] [-maxIt n] initP [initP2] polyFileName

By default the program uses bisection, but this can be modified with -newt for Newton's or -sec for Secant. The program should attempt 10,000 iterations by default, but this can be modified with -maxIter and the number of desired iterations. The initial point is provided (or an extra point for bisection and secant). For example, to run bisection method on file fun1.pol, with initial points 0 and 1:

> polRoot 0 1 fun1.pol

to run newton's with initial point 0:

> polRoot -newt 0 fun1.pol

and to run secant, with initial points 0 and 1, for 100,000 iterations:

> polRoot -sec -maxIter 100000 0 1 fun1.pol

'''
def parse_args():
    '''
    get filename, method, max iterations, initial points 
    '''
    args = sys.argv[1:]

    method = "bisection"
    max_iter = 10000

    if "-newt" in args:
        method = "newton"
    elif "-sec" in args:
        method = "secant"
    elif "-hybrid" in args:
        method = "hybrid"
    
    if "-maxIter" in args:
        i = args.index("-maxIter")
        max_iter = int(args[i+1])

    filename = None
    for arg in args:
        if arg.endswith(".pol"):
            filename = arg

    if filename is None:
        print("error: missing .pol file")
        sys.exit(1)

    nums = []
    for arg in args:
        try:
            nums.append(float(arg))
        except:
            continue

    return method, max_iter, nums, filename

#File i/o helpers
def read_polynomial(filename):
    '''
    reads n and coefficients
    returns coefficents
    '''
    with open(filename, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    coefficents = [float(x) for x in lines[1].split()]

    return coefficents

def write_solution(filename, root, iterations, success):
    '''
    write root, iterations, outcome
    '''
    out_file = filename.replace(".pol", ".sol")
    outcome = "success" if success else "fail"

    with open(out_file, "w") as f:
        f.write(f"{root} {iterations} {outcome}")

#polynomial helper
def evaluate_polynomial(coefficents, x):
    '''
    evaluates polynomial at x
    '''
    result = 0
    for c in coefficents:
        result = (result*x) + c
    return result

#bisection helpers
def compute_midpoint(left, right):
    '''
    midpoint of interval
    '''
    return (left+right) / 2

def check_interval(f_left, f_right):
    '''
    checks if interval contains a root
    '''
    return f_left * f_right < 0

def check_convergence(left, right, f_mid, eps):
    '''
    stopping condition
    '''
    return abs(right-left)/2 < eps or abs(f_mid) < eps

#bisection method
def bisection(coefficents, left, right, max_iter, eps):
    '''
    bisection method 
    '''
    f_left = evaluate_polynomial(coefficents, left)
    f_right = evaluate_polynomial(coefficents, right)

    if abs(f_left) < eps:
        return left, 0, True

    if abs(f_right) < eps:
        return right, 0, True

    if not check_interval(f_left, f_right):
        print("invalid values")
        return left, 0, False
    
    for iteration in range(1, max_iter+1):
        mid = compute_midpoint(left, right)
        f_mid = evaluate_polynomial(coefficents, mid)

        if check_convergence(left, right, f_mid, eps):
            return mid, iteration, True

        if f_left*f_mid < 0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid

    return mid, max_iter, False

#newton's helper
def check_small_slope(derivative, delta):
    '''
    checks if derivative is too small
    '''
    return abs(derivative) < delta

def evaluate_derivative(coefficents, x):
    '''
    evaluates derivative of polynomial at x
    '''
    result = 0
    n = len(coefficents)

    for i in range(n-1):
        power = n-i-1
        result = (result*x) + (power*coefficents[i])

    return result

#newton's
def newton(coefficents, intial, max_iter, eps, delta):
    '''
    Newton's method
    '''
    current = intial

    for iteration in range(1, max_iter+1):
        derivative_value = evaluate_derivative(coefficents, current)
        f_value = evaluate_polynomial(coefficents, current)

        if check_small_slope(derivative_value, delta):
            print("small slope")
            return current, iteration, False

        step = f_value / derivative_value
        next_value = current-step

        if abs(step) < eps:
            return next_value, iteration, True

        current = next_value

    return current, max_iter, False

#secant
def secant(coefficents, first_point, second_point, max_iter, eps):
    '''
    secant method
    '''
    x0 = first_point
    x1 = second_point

    f0 = evaluate_polynomial(coefficents, x0)
    f1 = evaluate_polynomial(coefficents, x1)

    for iteration in range(1, max_iter+1):
        if f1-f0 == 0:
            return x1, iteration, False

        x2 = x1 - ((f1*(x1-x0)) / (f1-f0))

        if abs(x2-x1) < eps:
            return x2, iteration, True

        # shift forward
        x0 = x1
        f0 = f1

        x1 = x2
        f1 = evaluate_polynomial(coefficents, x1)

    return x1, max_iter, False

#hybrid
def hybrid(coefficents, left, right, max_iter, eps, delta):
    '''
    hybrid method: bisection then newton
    '''
    #bisection
    root, iterations, _ = bisection(coefficents, left, right, 10, eps)

    #newton
    new_root, new_iterations, success = newton(
        coefficents, root, max_iter-iterations, eps, delta
    )

    return new_root, iterations + new_iterations, success

#Main
def polRoot():
    '''
    main function:
    read input
    solve
    write output
    '''
    method, max_iter, nums, filename = parse_args()
    coeffs = read_polynomial(filename)

    eps = np.finfo(np.float32).eps
    delta = eps

    if method == "bisection":
        if len(nums) < 2:
            print("error: need 2 intital points")
            sys.exit(1)
        left = nums[-2]
        right = nums[-1]
        root, iterations, success = bisection(coeffs, left, right, max_iter, eps)

    elif method == "newton":
        if len(nums) < 1:
            print("error: need 1 intital point")
            sys.exit(1)
        root, iterations, success = newton(coeffs, nums[-1], max_iter, eps, delta)

    elif method == "secant":
        if len(nums) < 2:
            print("error: need 2 intital points")
            sys.exit(1)
        root, iterations, success = secant(coeffs, nums[-2], nums[-1], max_iter, eps)

    elif method == "hybrid":
        if len(nums) < 2:
            print("error: need 2 intital points")
            sys.exit(1)
        root, iterations, success = hybrid(coeffs, nums[-2], nums[-1], max_iter, eps, delta)

    write_solution(filename, root, iterations, success)

if __name__ == "__main__":
    polRoot()