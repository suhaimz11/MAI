import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import expit as sigmoid

plt.ion()
np.random.seed(42)
SAMPLES_PER_CLASS = 30

#data
class1X = np.random.multivariate_normal(mean = [-1.5,-1], cov = [[+1.0,+0.0],[+0.0,+1.0]], size = SAMPLES_PER_CLASS)
class0X = np.random.multivariate_normal(mean = [+1.0,+1], cov = [[+1.5,+0.0],[+0.0,+1.5]], size = SAMPLES_PER_CLASS)
X = np.concatenate((np.ones((2*SAMPLES_PER_CLASS,1)),np.concatenate((class1X,class0X))),1)
y = np.concatenate((np.ones(SAMPLES_PER_CLASS),-np.ones(SAMPLES_PER_CLASS)))[:,None]

#model
class Model:
    def __init__(self,X,y,reg,bw):
        #TODO 1: initialize 
        # X [shape: (2*SAMPLES_PER_CLASS,3)-array]
        # y [shape: (2*SAMPLES_PER_CLASS,1)-array]
        # alpha [shape: (2*SAMPLES_PER_CLASS,1)-array]
        # K [shape: (2*SAMPLES_PER_CLASS,2*SAMPLES_PER_CLASS)-array]
        # reg [shape: scalar -- *NOT* (1,1)-array]
        self.X = X; self.y = y; 
        self.alpha = np.random.rand(self.X.shape[0],1)
        self.kernel = self.makeGaussianKernel(bw)
        self.K = self.kernel(self.X,self.X)
        self.reg = reg

    #returns a function that takes two matrices and produces the RBF kernel matrix between their rows.
    def makeGaussianKernel(self,bw): return lambda z1,z2: np.exp(-np.sum((z1[:,None,:]-z2[None,:,:])**2,2)/(2*bw**2))
    
    def loss(self): 
        #TODO 2: implement kernel SVM loss [returns (1,1)-array -- *NOT* scalar]
        return np.ones((1,self.X.shape[0]))*np.maximum(0,1-self.y*(self.K @ self.alpha)) + (self.reg/2)*(self.alpha.T @ self.K @ self.alpha)
    
    def lossGradient(self):  
        #TODO 3: implement kernel SVM loss gradient [returns (2*SAMPLES_PER_CLASS,1)-array]
        return -self.K @ (self.y*((1-self.y*(self.K @ self.alpha))>0).astype(float)) + self.reg*(self.K @ self.alpha)
    
    def gradientStep(self,gamma): 
        #TODO 4: perform a gradient step, i.e., use self.lossGradient to update self.alpha [returns None]
        self.alpha -= gamma*self.lossGradient()

    def predict(self,X): 
        #TODO 5: make a prediction for the (t,3)-array X [returns (2*SAMPLES_PER_CLASS,1)-array]
        return self.kernel(X,self.X) @ self.alpha

    def plot(self, ax, prob=False, resolution=0.02):

        # define bounds
        x_min, x_max = self.X[:,1].min()-1, self.X[:,1].max()+1
        y_min, y_max = self.X[:,2].min()-1, self.X[:,2].max()+1
        
        # predict mesh grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),np.arange(y_min, y_max, resolution))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = self.predict(np.concatenate((np.ones((grid.shape[0],1)),grid),1)).reshape(xx.shape)
        zz = sigmoid(zz) if prob else np.sign(zz)

        # plot regions
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['red', 'blue'])
        ax.contourf(xx, yy, zz, alpha=0.3, levels=np.linspace(0,1,21) if prob else [-1,0,1], cmap=cmap, vmin=0 if prob else -1, vmax=1)
        ax.contour (xx, yy, zz, alpha=0.7, levels=([0.5] if prob else [0.0]), colors='k', linewidths=1)

        # plot data
        ax.scatter(self.X[self.y[:,0]==-1,1], self.X[self.y[:,0]==-1,2], c='red', label='-1')
        ax.scatter(self.X[self.y[:,0]==+1,1], self.X[self.y[:,0]==+1,2], c='blue', label='+1')

        # plot labels
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        ax.set_title("kernel SVM")
        ax.legend()

#optimization
iter = 0 #gradient descent iteration
gamma = 1e-3 #learning rate
eps = 1e-3 #stopping threshold
reg = 1 #regularization parameter
bw = 1 #rbf bandwidth
model = Model(X,y,reg,bw)

fig, ax = plt.subplots()
while norm(model.lossGradient())>eps: #run until loss gradient norm drops below eps
    iter += 1
    #TODO 6: execute a gradient step with learning rate gamma
    model.gradientStep(gamma)
    print("Iteration: "  + '{:04.0f}'.format(iter) + "  ||  Loss: " + '{:+07.2f}'.format(model.loss()[0,0])  + "   ||   Norm of loss gradient: " + '{:+07.2f}'.format(norm(model.lossGradient())))
    
    #visualization
    ax.clear()    
    model.plot(ax,True)
    fig.canvas.draw()
    fig.canvas.flush_events()