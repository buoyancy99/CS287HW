"""
Author: Pieter Abbeel pabbeel@cs.berkeley.edu www.cs.berkeley.edu/~pabbeel 
2009/11/07; updated 2015/9/1

Python edition: Boyuan Chen buoyuan99@gmail.com, Harry Xu xuhuazhe12@gmail.com
latest update 2019/7/21


cvx_intro.py
CVXPY is a Python-embedded modeling language for convex optimization problems. 
It automatically transforms the problem into standard form, calls a solver, 
and unpacks the results. CVXPY's power comes from being able to very conveniently 
write down the convex optimization problem, i.e., at a relatively high level of
abstraction


To obtain CVXPY: follow the download and install instructions:
https://www.cvxpy.org/install/index.html
"""


#There are numerous examples on the above website.  Let's consider just a
#few here to give you a concrete starting point.

import cvxpy as cp
import numpy as np

def Example_1():
    print("\n=======================================================")
    print("Example 1 solves a simple optimization problem in CVXPY:")

    # Create two scalar optimization variables.
    x = cp.Variable()
    y = cp.Variable()

    # Create two constraints.
    constraints = [x + y == 1,
                x - y >= 1]

    # Form objective.
    obj = cp.Minimize((x - y)**2)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value, y.value)


def Example_2():
    print("\n=======================================================")
    print("Example 2 solves a infeasible problems in CVXPY:")

    x = cp.Variable()

    # An infeasible problem.
    prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)

    # An unbounded problem.
    prob = cp.Problem(cp.Minimize(x))
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)

def Example_3():
    print("\n=======================================================")
    print("Example 3 solves a constrained least squares problem Ax=b in CVXPY:")

    # Problem data.
    m = 10
    n = 5
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Construct the problem.
    x = cp.Variable(n)
    # Notice that in numpy you use * for elementwise multiplication 
    # However, in cvxpy * is matrix muliplication when two sizes are matricies or vectors
    # To do elementwise multiplication use cvxpy.multiply
    # You can still use the numpy @ operator for matmul as well, just substitute * with @ in next line
    objective = cp.Minimize(cp.sum_squares(A * x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)

    print("Optimal value", prob.solve())
    print("Optimal var")
    print(x.value) # A numpy ndarray.

def Example_4():
    print("\n=======================================================")
    print("Example 4 minimize the infinite-norm of Ax-b in CVXPY:")

    # Problem data.
    m = 10
    n = 5
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Construct the problem. 
    x = cp.Variable(n)
    # Notice that in numpy you use * for elementwise multiplication 
    # However, in cvxpy * is matrix muliplication when two sizes are matricies or vectors
    # To do elementwise multiplication use cvxpy.multiply
    # You can still use the numpy @ operator for matmul as well, just substitute * with @ in next line
    objective = cp.Minimize(cp.norm_inf(A * x - b))
    prob = cp.Problem(objective)

    print("Optimal value", prob.solve())
    print("Optimal var")
    print(x.value) # A numpy ndarray.


def Example_5():
    print("\n=======================================================")
    print("Example 5 solves the standard hard-margin SVM problem in CVXPY:")
    print("For most random X and y, the above problem will be infeasible.")
    print("Change random seed in the example to seek working seeds")

    # Problem data.
    m = 10
    n = 5
    np.random.seed(1)

    # rows of X are the feature vectors
    # rows of y are the labels (-1 or +1)
    # find the minimum norm w such that the feature vectors with positive label are separated by
    # a margin greater than or equal to 2 from the feature vectors with negative label
    X = np.random.randn(m, n)
    y = np.random.randint(2, size=m)

    # Construct the problem. 
    w = cp.Variable(n)
    b = cp.Variable()

    objective = cp.Minimize(cp.sum_squares(w) / 2)
    constraints = [cp.multiply((X * w + b) , y) >= 1]
    prob = cp.Problem(objective, constraints)

    print("Optimal value", prob.solve())
    print("Optimal var")
    print('w:', w.value, '\nb:', b.value)

def Example_6():
    print("\n=======================================================")
    print("Example 6 solves the standard soft-margin SVM problem in CVXPY:")

    # Problem data.
    m = 10
    n = 5
    np.random.seed(1)

    C = 1
    # rows of X are the feature vectors
    # rows of y are the labels (-1 or +1)
    # find the minimum norm w such that the feature vectors with positive label are separated by
    # a margin greater than or equal to 2 from the feature vectors with negative label
    X = np.random.randn(m, n)
    y = np.random.randint(2, size=m)

    # Construct the problem. 
    w = cp.Variable(n)
    xi = cp.Variable(m)
    b = cp.Variable()

    objective = cp.Minimize(cp.sum_squares(w) / 2 + C * cp.sum(xi))
    constraints = [cp.multiply((X * w + b) , y) >= 1 - xi, xi >= 0]
    prob = cp.Problem(objective, constraints)

    print("Optimal value", prob.solve())
    print("Optimal var")
    print('w:', w.value, '\nb:', b.value)



if __name__ == "__main__":

    Example_1()
    """
    Output:
    status: optimal
    optimal value 1.0
    optimal var 1.0 1.570086213240983e-22
    """

    Example_2()
    """
    Output:
    status: infeasible
    optimal value inf
    status: unbounded
    optimal value -inf
    """

    Example_3()
    """
    Output:
    Optimal value 4.141338603672535
    Optimal var [-4.95922264e-21  6.07571976e-21  1.34643668e-01  1.24976681e-01  -4.57130806e-21]
    """

    Example_4()
    """
    Output:
    Optimal value 0.774925360074766
    Optimal var [-0.17266601 -0.01055959  0.01675094  0.11737575 -0.07621592]
    """

    Example_5()
    """
    Output:
    Optimal value inf
    Optimal var
    w: None 
    b: None
    """

    Example_6()
    """
    Output:
    Optimal value 4.999289866842231
    Optimal var
    w: [ 4.47318986e-13 -1.47228832e-13  4.72408221e-13  2.39124801e-13
    -2.74240749e-13] 
    b: 1.6000635061213848
    """
