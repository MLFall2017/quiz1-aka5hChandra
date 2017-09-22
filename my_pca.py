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
       
    def findPCA(self):
        self.mean = np.mean(self.data,0)
        meanCenteredData = (self.data - self.mean)
        self.cov = (meanCenteredData.T.dot(meanCenteredData)) / (self.data.shape[0]-1) 
        self.eigenValues ,self.eigenVectors = np.linalg.eig(self.cov)
        ids = np.argsort(-1 * self.eigenValues)
        self.eigenValues = self.eigenValues[ids]
        self.eigenVectors = self.eigenVectors[ids , :]
        print("Eigan Values are")
        print( self.eigenValues )
        print("\nEigan Vectors (Principle Components)are")
        print(self.eigenVectors)


    def findVariance(self, x):
        meanX = np.mean(x,0)
        meandCenteredX = x - meanX
        variance = (meandCenteredX.dot(meandCenteredX)) / (meandCenteredX.shape[0]-1) 
        return variance

    def findCOVariance(self, x , y):
        meanX = np.mean(x,0)
        meandCenteredX = x - meanX

        meanY = np.mean(y,0)
        meandCenteredY = y - meanY
        coVariance = (meandCenteredX.T.dot(meandCenteredY)) / (meandCenteredX.shape[0]-1) 
        return coVariance
    
    def findAllVariances(self):
        
        print("Variance of x is %f"%(self.findVariance(self.data[:,0]))) 
        print("Variance of y is %f"%(self.findVariance(self.data[:,1]))) 
        print("Variance of z is %f"%(self.findVariance(self.data[:,2])))

    def findAllCOVariances(self):
        print("co-Variance of x and yis %f"%(self.findCOVariance(self.data[:,0],self.data[:,1])))
        print("co-Variance of y and z is %f"%(self.findCOVariance(self.data[:,1],self.data[:,2])))
      
       
    
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
    pca.findAllVariances()
    pca.findAllCOVariances()
    pca.findPCA()
    pca.projectToEigenSpace()
    #pca.printVarianceOfEachColumn()
 