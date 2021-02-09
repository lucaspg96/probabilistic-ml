import numpy as np
import matplotlib.pyplot as plt

class PPCA():
    
    def __init__(self, n_components = 2):
        self.L = n_components
        
    def log_likelihood(self, X, mean, W, M_inv, sigma, L):
        N,D = X.shape
        s = 0
        for x in X:
            z =  M_inv.dot(W.T).dot((x - mean))

            s += (D/2)*np.log(2*np.pi*sigma) + (L/2)*np.log(2*np.pi) + 0.5*np.trace(np.outer(z,z))
            s += (1/(2*sigma))*np.linalg.norm(x - mean)**2 - (1/sigma)*z.dot(W.T).dot(x-mean)
            s += 1/(2*sigma)*np.trace(np.outer(z,z).dot(W.T).dot(W))

        return -s

    def fit(self, X, W_0 = None, sigma_0=1, iter=10, debug=False):
        N,D = X.shape
        mean = np.mean(X, axis=0)
        
        L = self.L
        W = W_0.copy() if not W_0 is None else np.random.randn(D, L)
        sigma = sigma_0
        M_inv = None

        log_likelihoods = []

        for i in range(iter):
            M_inv = np.linalg.inv(W.T.dot(W) + sigma*np.eye(L))

            e_z = M_inv.dot(W.T).dot((X - mean).T)
            e_zz = sigma*M_inv + e_z.dot(e_z.T)

            W = ((X - mean).T.dot(e_z.T)).dot(np.linalg.inv(e_zz))

            s = 0
            for i,x in enumerate(X):
                e_zi = e_z[:,i]
                e_zzi = sigma*M_inv + np.outer(e_zi, e_zi) 
                s += np.linalg.norm(x-mean)**2 - 2*e_zi.T.dot(W.T).dot(x-mean) + np.trace(e_zzi.dot(W.T).dot(W))

            sigma = s/(N*D)  

            if debug:
                log_likelihoods.append(self.log_likelihood(X, mean, W, M_inv, sigma, L))

            if debug:
                print("\n-----------------------------------\n")
                print(f"Iteração {i+1}")
                print(f"W:\n{W}")
                print(f"Sigma: {sigma}")
                print(f"Log-verossimilhança: {log_likelihoods[-1]}")
        if debug:
            plt.plot(log_likelihoods, label="Log-verossimilhança do treinamento")
            plt.legend()
            plt.show()
        
        self.mean = mean
        self.W = W
        self.sigma = sigma
        self.M_inv = M_inv
        
    def project(self, X):
        return self.M_inv.dot(self.W.T).dot((X - self.mean).T).T
    
    def reconstruct(self, z):
        return self.W.dot(z.T).T + self.mean
    
    

class LogisticRegression():
    
    def __init__(self):
        pass
    
    def logistic_function(self,a):
        return 1/(1+np.exp(-1*a))

    def R(self,x,w):
        prediction = self.logistic_function(x.dot(w))
        return np.diag(np.multiply(prediction, (1-prediction)).ravel())

    def A(self,x,w,S_inv):
        r = self.R(x,w)
        return x.T.dot(r).dot(x) + S_inv

    def IRLS_map(self,x,y, m, S, tol=1e-3, maxiter=100):
        w = np.zeros(x.shape[1]).reshape(-1,1)
        previous_w = None
        i = 0
        S_inv = np.linalg.inv(S)
        m = m.reshape(-1,1)
        y = y.reshape(-1,1)

        while i < maxiter and (previous_w is None or not all(np.isclose(w,previous_w, rtol=tol))):
            previous_w = np.copy(w)
            a = self.A(x,w,S_inv)
            prediction = self.logistic_function(x.dot(w))

            aux = x.T.dot(y-prediction) - S_inv.dot(w - m)

            w = w + np.linalg.inv(a).dot(aux)
            i += 1

        return w.ravel()

    def fit(self,x,y, m=None, S=None):
        m = m if not m is None else np.zeros(x.shape[1])
        S = S if S is not None else np.eye(m.shape[0])
        
        w_hat = self.IRLS_map(x,y,m,S)
        S_inv = np.linalg.inv(S)
        r_hat = self.R(x,w_hat)
        H = x.T.dot(r_hat).dot(x) + S_inv
        
        self.w_hat = w_hat
        self.var = np.linalg.inv(H)
        
    def predict_probit(self,x):
        m_a = x.dot(self.w_hat)
        sigma_a = x.dot(self.var).dot(x.T)

        y = self.logistic_function(np.sqrt(1 + np.pi*sigma_a/8).dot(m_a))

        return np.round(np.nan_to_num(y, 0))