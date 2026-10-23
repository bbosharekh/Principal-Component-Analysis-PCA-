class PCA:

    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None


    def fit(self, X):
        # mean
        self.mean=np.mean(X,axis=0)
        X = X - self.mean
        
        # covariance 
        # row = 1 sample, columns = featurs
        cov = np.cov(X.T)
        
        # eigenvectors, eigenvalues 
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # sort eigenvectors
        eigenvectors = eigenvectors.T # since they are column vectors 
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]
        eigenvalues = eigenvalues[idxs]
        
        #store the first n eigenvectors 
        self.components = eigenvectors[0:self.n_components].T
        
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()
        return self
        
    def transform(self,X):
        # project data 
        X = X - self.mean
        return np.dot(X,self.components) 