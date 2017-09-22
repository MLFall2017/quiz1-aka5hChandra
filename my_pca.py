import numpy as np 
import matplotlib.pyplot as plt

class my_pca:
    data = []
    mean = 0
    cov = 0
    eigenValues = []
    eigenVectors = []
    projectedData = []
    def __init__(self , csvPath):
        self.data = np. genfromtxt(csvPath, delimiter=',', dtype = 'U5')
        self.header = self.data[0,:]
        self.data = self.data[1:,:].astype(np.float32)
        self.mean = np.mean(self.data,0)
        meanCenteredData = (self.data - self.mean)
        self.cov = (meanCenteredData.T.dot(meanCenteredData)) / (self.data.shape[0]-1) 
        self.eigenValues ,self.eigenVectors = np.linalg.eig(self.cov)
        ids = np.argsort(-1 * self.eigenValues)
        self.eigenValues = self.eigenValues[ids]
        self.eigenVectors = self.eigenVectors[ids , :]
    
    def projectToEigenSpace(self):
        '''
        Projects input data to Eigen Space (Principal Component Space)
        '''
        self.projectedData = self.data.dot(self.eigenVectors)   
    
    def printVarianceOfEachColumn(self):
        '''
        Prints Variance of Each input data 
        '''
        vairance = np.diagonal(self.cov)
        for id , var in enumerate(vairance):
             print("Variance of %s is %f"%(self.header[id], var))

if __name__ == "__main__": 
    dataPath =   input("This is simple PCA Module\nEnter path to data file\n")#'D:\\UNCC\\ML\quiz _1\\quiz1-aka5hChandra\\dataset_1.csv'#
    pca =  my_pca(dataPath)
    pca.projectToEigenSpace()
    pca.printVarianceOfEachColumn()
 