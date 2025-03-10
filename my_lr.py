def A(x, y):
    top_part = sum(xi*yi for xi, yi in zip(x,y)) - sum(x) * sum(y)
    bottom_part = sum(xi**2 for xi in x) - (sum(x))**2
    return top_part/bottom_part

def B(x, y):
    n = len(x)
    return (1/n) * sum(y) - A(x, y) * (1/n) * sum(x)

def linear_regression(pred_x, x, y):
    slope = A(x, y)
    intercept = B(x, y)
    return intercept + slope * pred_x