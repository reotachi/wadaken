import numpy as np
from scipy.stats import t

def akane(y, x):
    
    law_x = x.shape
    N, k = law_x[1], law_x[0]
    
    beta_hat = np.linalg.inv(x @ x.T) @ x @ y  # OLS estimate beta_hat
    yhat = x.T @ beta_hat # fitted value
    uhat = y - yhat # residual
    s2 = (uhat.T @ uhat) / (N - k) # estimated variance of the error term 
    u2 = uhat ** 2

    # homoskedasticity-only standard errors
    var = np.diag(s2 * np.linalg.inv(x @ x.T))
    if np.all(var > 0):
        se = var ** 0.5
    else:
        print("error: homoskedasticity-only variance is less than zero")
        ans = [beta_hat, yhat, uhat, s2, u2]
        return ans
    
    # # heteroskedasticity-robust standard errors
    var = np.diag((np.linalg.inv(x @ x.T) @ (N / (N - k) * x @ np.diag(u2) @ x.T) @ np.linalg.inv(x @ x.T)))
    if np.all(var > 0):
        se = var ** 0.5
    else:
        print("error: heteroskedasticity-robust variance is less than zero")
        ans = [beta_hat, yhat, uhat, s2, u2]
        return ans
    
    # % t-statistic
    tval = beta_hat / se
    
    # p-value
    pval = []
    for i in range(len(beta_hat)):
        pval.append(2 * (1 - t.cdf(x = abs(tval[i]), df = N - k)))

    # confidence interval
    lower = []
    upper = []
    for i in range(len(beta_hat)):
        lower.append(t.interval(0.95, N - 1, beta_hat[i], se[i])[0])
        upper.append(t.interval(0.95, N - 1, beta_hat[i], se[i])[1])
    
    # R-squared   ESS/TSS or 1-SSR/TSS
    R2 = 1 - (uhat.T @ uhat) / ((y - np.mean(y)).T @ (y - np.mean(y)))
    
    #degrees of freedom adjusted R2 
    R2adj = 1 - ( N - 1) / (N - k) * (uhat.T @ uhat) / ((y - np.mean(y).T) @ (y - np.mean(y)))
    
    ans = [beta_hat, yhat, uhat, s2, u2, se, tval, pval, lower, upper, R2, R2adj]
    
    return (ans)