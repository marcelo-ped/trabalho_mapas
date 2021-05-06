from __future__ import print_function
#!-*- conding: utf8 -*-
import os

def make_all_csv():
    num_dat = 0
    file_dat = "/home/marcelo/Desktop/testtt/new/TE_data_me" + str(num_dat) + ".dat"
    lines_list = []
    for i in range(11):
        num_dat = num_dat + 1
        file_dat = "fault_1/TE_data_me" + "{:02d}".format(i + 1) + ".dat"
        with open(file_dat, "r") as reader:
            lines = reader.readlines()
            lines_list.append(lines)
    l = ""
    j = 0
    while j <= 959:
        with open('fault_1.csv', 'a') as csvfile:
            for i in range(11):
                if i < 10:
                    l = l + str(lines_list[i][j]).replace('\n', '')
                else:
                    l = l + str(lines_list[i][j])
                csvfile.write(l)
                l = ""
        j = j + 1
    del(lines_list[:])
    for i in range(11):
        num_dat = num_dat + 1
        file_dat = "fault_2/TE_data_me" + "{:02d}".format(i + 1) + ".dat"
        with open(file_dat, "r") as reader:
            lines = reader.readlines()
            lines_list.append(lines)
    l = ""
    j = 0
    while j <= 959:
        with open('fault_2.csv', 'a') as csvfile:
            for i in range(11):
                if i < 10:
                    l = l + str(lines_list[i][j]).replace('\n', '')
                else:
                    l = l + str(lines_list[i][j])
                csvfile.write(l)
                l = ""
        j = j + 1
"""
========================================
VISUALIZE TENNESSEE EASTMAN VARIABLES
========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.manifold import TSNE
import SimpSOM as sps
from sklearn.cluster import KMeans
from minisom import MiniSom

'''
import sys
pcadir = '/home/thomas/Dropbox/posgrad/patternrecog/soft/PCA'
sys.path.append(pcadir)
from pca import eigen
'''
# Python numpy.linalg.eig does not sort the eigenvalues and eigenvectors
def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)
    idx = idx[::-1] # Invert from ascending to descending
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)


class TE():
    """ Tennessee Eastman Simulator Data Reading and Manipulation
    Parameters
    ----------

    Attributes
    ----------
    Each training data file contains 480 rows and 52 columns and
    each testing data file contains 960 rows and 52 columns.
    An observation vector at a particular time instant is given by

    x = [XMEAS(1), XMEAS(2), ..., XMEAS(41), XMV(1), ..., XMV(11)]^T
    where XMEAS(n) is the n-th measured variable and
    XMV(n) is the n-th manipulated variable.
    """
    
    XMEAS = ['Input Feed - A feed (stream 1)'	,       	#	1
        'Input Feed - D feed (stream 2)'	,       	#	2
        'Input Feed - E feed (stream 3)'	,       	#	3
        'Input Feed - A and C feed (stream 4)'	,       	#	4
        'Miscellaneous - Recycle flow (stream 8)'	,	#	5
        'Reactor feed rate (stream 6)'	,                 	#	6
        'Reactor pressure'	,                           	#	7
        'Reactor level'	,                                	#	8
        'Reactor temperature'	,                           	#	9
        'Miscellaneous - Purge rate (stream 9)'	,       	#	10
        'Separator - Product separator temperature'	,	#	11
        'Separator - Product separator level'	,       	#	12
        'Separator - Product separator pressure'	,	#	13
        'Separator - Product separator underflow (stream 10)'	,	#	14
        'Stripper level'	,                           	#	15
        'Stripper pressure'	,                           	#	16
        'Stripper underflow (stream 11)'             	,	#	17
        'Stripper temperature'	,                           	#	18
        'Stripper steam flow'	,                           	#	19
        'Miscellaneous - Compressor work'	,       	#	20
        'Miscellaneous - Reactor cooling water outlet temperature'	,	#	21
        'Miscellaneous - Separator cooling water outlet temperature'	,	#	22
        'Reactor Feed Analysis - Component A'	,	#	23
        'Reactor Feed Analysis - Component B'	,	#	24
        'Reactor Feed Analysis - Component C'	,	#	25
        'Reactor Feed Analysis - Component D'	,	#	26
        'Reactor Feed Analysis - Component E'	,	#	27
        'Reactor Feed Analysis - Component F'	,	#	28
        'Purge gas analysis - Component A'	,	#	29
        'Purge gas analysis - Component B'	,	#	30
        'Purge gas analysis - Component C'	,	#	31
        'Purge gas analysis - Component D'	,	#	32
        'Purge gas analysis - Component E'	,	#	33
        'Purge gas analysis - Component F'	,	#	34
        'Purge gas analysis - Component G'	,	#	35
        'Purge gas analysis - Component H'	,	#	36
        'Product analysis -  Component D'	,	#	37
        'Product analysis - Component E'	,	#	38
        'Product analysis - Component F'	,	#	39
        'Product analysis - Component G'	,	#	40
        'Product analysis - Component H']		#	41
			
    XMV = ['D feed flow (stream 2)'	,                 	#	1 (42)
        'E feed flow (stream 3)'	,                 	#	2 (43)
        'A feed flow (stream 1)'	,                 	#	3 (44)
        'A and C feed flow (stream 4)'	,                 	#	4 (45)
        'Compressor recycle valve'	,                 	#	5 (46)
        'Purge valve (stream 9)'	,                 	#	6 (47)
        'Separator pot liquid flow (stream 10)'	,       	#	7 (48)
        'Stripper liquid product flow (stream 11)'	,	#	8 (49)
        'Stripper steam valve'	,                           	#	9 (50)
        'Reactor cooling water flow'	,                 	#	10 (51)
        'Condenser cooling water flow'	,                 	#	11 (52)
        'Agitator speed']             # constant 50%			12 (53)

    def var_category_str(self, featnr):
        '''Returning string with the original category 'XMEAS #' or 'XMV #'
        '''
        if featnr < 41:
            name = 'XMEAS (' + str(featnr+1) + '): '
        else:
            name = 'XMV (' + str(featnr+1-41) + '): '
        return name


    def __init__(self):
        #print('Executing __init__() ....')

        self.Xtrain = None
        self.Xtest = None
        self.featname = self.XMEAS + self.XMV
        self.extendedfeatname = list(self.featname)
        self.numfeat = len(self.featname)
        for i in range(self.numfeat):
            self.extendedfeatname[i] = self.var_category_str(i) + self.featname[i]
        #print('TE.extendedfeatname=', self.extendedfeatname);
        #print('TE.featname=', self.featname); quit()

    def standardize(self):
        print('Data standardization to zero mean and unit variance...')
        X = self.Xtrain
        #print('\nTraining dataset before standardization=\n', X)
        #print('\nTest dataset before standardization=\n', self.Xtest)
        self.meanX = np.mean(X, axis=0)
        # ddof=1 ==> divide by (n-1) --- ddof=0 ==> divide by n
        ddof_std = 0    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.std.html#numpy.std
        self.stdX = X.std(axis=0, ddof=ddof_std)

        #print('Dataset statistic:\n Mean=', meanX, '\nStandard deviation=\n', stdX)
        #minX = X.min(axis=0)
        #maxX = X.max(axis=0)
        #print('Dataset statistic:\nMin=', minX, '\nMax=', maxX )

        self.Xcentered_train = X - self.meanX
        #print('Dataset X=\n', X, '\nDataset centralized Xcentered_train=\n', self.Xcentered_train)
        self.Xstandardized_train = self.Xcentered_train / self.stdX
        #print('Dataset standadized Xstandardized_train=\n', self.Xstandardized_train)

        self.Xcentered_test = self.Xtest - self.meanX
        self.Xstandardized_test = self.Xcentered_test / self.stdX 


    def labelledcsvread(self, filename, delimiter = '\t', fmode='r'):

        print('Reading CSV from file ', filename )
        f = open(filename, fmode)
        reader = csv.reader(f, delimiter=delimiter)
        ncol = len(next(reader)) # Read first line and count columns
        nfeat = ncol-1
        f.seek(0)              # go back to beginning of file
        #print('ncol=', ncol); quit()
        
        x = np.zeros(nfeat)
        X = np.empty((0, nfeat))
        y = []
        for row in reader:
            #print(row)
            for j in range(nfeat):
                x[j] = float(row[j])
                #print('j=', j, ':', x[j])
            X = np.append(X, [x], axis=0)
            label = row[nfeat]
            y.append(label)
            #print('label=', label)
            #quit()
        #print('X.shape=\n', X.shape)#, '\nX=\n', X)
        #print('y=\n', y)
        
        
        # Resubsitution for all methods
        from sklearn.preprocessing import LabelEncoder
        from LabelBinarizer2 import LabelBinarizer2
        lb = LabelBinarizer2()
        Y = lb.fit_transform(y)
        classname = lb.classes_
        #print('lb.classes_=', lb.classes_, '\nY=\n',Y)

        le = LabelEncoder()
        ynum = le.fit_transform(y)
        #print(ynum)
        
        return X, Y, y, ynum, classname


    def read_train_test_pair(self, datadir, fault_num='01', standardize=True):
        ''' Read a training and test pair from the predefined TE datasets and put them
        into the respective data structures
        '''
        ftrain = datadir+'d'+fault_num+'.dat'
        ftest = datadir+'d'+fault_num+'_te.dat'
        self.Xtrain = self.datacsvreadTE(ftrain)
        self.Xtest = self.datacsvreadTE(ftest)
        if standardize:
            self.standardize()

    def plot_condition(self, X, y, classlabel, classname, featname, plot_time_axis=True, time_offset=True,
            dropfigfile=None, title=None):
        '''Given a set of patters with class label, plot in 2D.
        If the time axis option is true, plot the postion in time, following
        the order in the data matrix X (first pattern X[0] at t=0
        '''
        if plot_time_axis:
            print ('Generating 2-D plot with time evolution ...')
        else:
            print ('Generating 2-D plot ...')
        #print('X=\n', X.shape, '\nclassname=', classname)
        numclasses = len(classname)
        xlab = featname[0]
        ylab = featname[1]
        fig, ax = plt.subplots(); # Create a figure and a set of subplots
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, numclasses)]

        #plot_time_axis=False   # DEBUG
        if plot_time_axis:
            from mpl_toolkits.mplot3d import Axes3D
            title = title + ' 2-D with time evolution'
            zlab = 't'
            ax = plt.figure().add_subplot(projection='3d', azim=-45, elev=30)
            print(ax)
            #ax = Axes3D(fig, azim=-45, elev=30)
            ax.set_title(title)
            ax.set_xlabel(xlab)
            #ax.w_xaxis.set_ticklabels([])
            ax.set_ylabel(ylab)
            #ax.w_yaxis.set_ticklabels([])
            ax.set_zlabel(zlab)
            #ax.w_zaxis.set_ticklabels([])
            toffset = 0
            for i in range(numclasses):
                idx = np.where(y == i)
                numpts = len(idx[0])
                t = np.linspace(toffset, toffset+numpts-1, numpts)
                if time_offset:
                    toffset += numpts
                ax.scatter(X[idx, 0], X[idx, 1], t, c=colors[i], label=classname[i])
            #ax.set_zlim(bottom=0, top=toffset+numpts)
        else:
            for i, color in zip(classlabel, colors):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, label=classname[i],
                        cmap=plt.cm.Paired, edgecolor='black', s=20)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.title(title)
        plt.legend()
        plt.axis('tight')
        if not dropfigfile is None:
            print('Saving figure in ', dropfigfile)
            plt.savefig(dropfigfile, dpi=1200)
        #plt.show()


    def plotscatter(self, datafile, feat1, feat2, standardize=True, dropfigfile=None,
            title='Tennessee Eastman: Classes in Feature Space'):
        delimiter = '\t'
        X, Y, y, ynum, classname = self.labelledcsvread(filename=datafile, delimiter=delimiter)
        #print('X=\n',X,'shape=',X.shape,'\nY=\n',Y,'shape=',Y.shape,'y=\n',y,'ynum=\n',ynum,'shape=',ynum.shape,'classname=',classname); quit()

        labels = ynum
        classes = classname
        classlabel = np.unique(ynum)

        X2feat = X[:, [feat1,feat2]] # only two features can be visualized directly

        X = X2feat
        y = ynum

        if standardize:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            #print('mean=', mean, 'std=', std)
            X = (X - mean) / std
        #featname = [self.featname[feat1], self.featname[feat2]]
        featname = [self.extendedfeatname[feat1], self.extendedfeatname[feat2]]
        self.plot_condition(X, y, classlabel, classname, featname, plot_time_axis=True,
                dropfigfile=None, title=title)


    def plot_train_test_pair(self, datadir, fault_num='01', feat1=1, feat2=2,
                             standardize=True, plot_time_axis=True, dropfigfile=None, title=None):
        ''' Plot the training and test pair
        '''
        self.read_train_test_pair(datadir=datadir, fault_num=fault_num, standardize=standardize)
        
        if standardize:
            Xtrain = self.Xstandardized_train
            Xtest = self.Xstandardized_test
        else:
            Xtrain = self.Xtrain
            Xtest = self.Xtest
        
        Xtrain = Xtrain[:, [feat1,feat2]]
        Xtest = Xtest[:, [feat1,feat2]]
        ntrain = Xtrain.shape[0]
        ntest = Xtest.shape[0]
        X = np.concatenate((Xtrain, Xtest), axis=0)
        
        y = np.concatenate((np.zeros(ntrain),np.ones(ntest)))
        classname = ['Normal', 'Fault'+' '+fault_num]
        classlabel = np.array([0, 1])
        featname = [self.extendedfeatname[feat1], self.extendedfeatname[feat2]]

        #print('ntrain=', ntrain, 'ntest=', ntest, 'n=', 'X=\n', X, '\nshape=', X.shape, 'y=\n',y,'shape=',y.shape,) #; quit()
        
        self.plot_condition(X, y, classlabel, classname, featname, plot_time_axis=plot_time_axis,
                dropfigfile=dropfigfile, title=title)
        
        

    def plot_train_test_pair_tSNE(self, datadir, fault_num='01',
                             standardize=True, plot_time_axis=True, dropfigfile=None, title=None):
        ''' Plot the training and test pair
        '''
        self.read_train_test_pair(datadir=datadir, fault_num=fault_num, standardize=standardize)
        
        if standardize:
            Xtrain = self.Xstandardized_train
            Xtest = self.Xstandardized_test
        else:
            Xtrain = self.Xtrain
            Xtest = self.Xtest
        from sklearn.manifold import TSNE
        n_components = 2
        X = np.concatenate((Xtrain, Xtest), axis=0)
        print('Generating tSNE plot...')
        X = TSNE(n_components=n_components).fit_transform(X)
        
        ntrain = Xtrain.shape[0]
        ntest = Xtest.shape[0]
        y = np.concatenate((np.zeros(ntrain),np.ones(ntest)))
        classname = ['Normal', 'Fault'+' '+fault_num]
        classlabel = np.array([0, 1])
        featname = ['tSNE 1', 'tSNE 2']
        
        self.plot_condition(X, y, classlabel, classname, featname, plot_time_axis=plot_time_axis,
                dropfigfile=dropfigfile, title=title)


    def plot_simultaneous(self, datadir, faults=('0','1', '2', '3'), method = 'tSNE', standardize=True,
            dropfigfile=None, title='Simultaneous Plot'):
        ''' Plot several conditions (faults) simultaneously
        '''
        n_components = 2
        num_faults = len(faults)
        for i in range(num_faults):
            fault_num = faults[i]
            ftest = '/home/marcelo/Desktop/TE_data_me01_' + fault_num + '.dat'
            Xi = self.datacsvreadTE(ftest)
            n, d = Xi.shape
            yi = np.ones(n)
            #print('Xi=\n', Xi, 'shape=', Xi.shape, 'i*yi=\n', i*yi, 'shape=', yi.shape)
            if i == 0:
                X = Xi
                y = np.zeros(n)
            else:
                X = np.concatenate((X, Xi), axis=0)
                y = np.concatenate((y, i * yi), axis=0)

        #print('X=\n', X, 'shape=', X.shape, 'y=\n', y, 'shape=', y.shape)
        classname = faults
        classlabel = np.array(range(num_faults))

        plot_time_axis = True
        if method == 'tSNE':
            print('Generating tSNE plot...')
            featname = ['tSNE 1', 'tSNE 2']
            X = TSNE(n_components=n_components).fit_transform(X)
        elif method == 'PCA':
            print('Generating PCA plot...')
            featname = ['PC 1', 'PC 2']
            meanX = np.mean(X, axis=0)
            Xcentered = X - meanX
            C = np.cov(X, rowvar=False) # Observations (samples) are the rows
            print('Covariance Matrix of Dataset C=\n', C)

            Lambda, PHI = eigen(C)
            print('Eigenvectors (=columns) of Covariance Matrix of Dataset PHI=\n', PHI)
            print('PHI * PHI\'=\n', np.dot(PHI, PHI.T))
            print('Eigenvalues of Covariance Matrix of Dataset Lambda=\n', Lambda)
            print('C * PHI=\n', np.dot(C, PHI))
            print('\nLAMBDA * PHI=\n', np.dot(np.diag(Lambda), PHI))

            X = np.dot(Xcentered, PHI)

        elif method == 'RadViz':
            print('Generating RadViz plot...')
            plot_time_axis = False
            import pandas as pd
            featname = self.extendedfeatname[:-1]

            # xxx mask=[0, 1, 2, 5, 10, 15, 20]
            X, featname = self.filter_vars(X, mask=[0, 1, 2, 5, 10, 15, 20])


            #print('X.shape=', X.shape, 'self.extendedfeatname.shape=', len(self.extendedfeatname[:-1]))
            df = pd.DataFrame(X, columns=featname)
            class_column_name = 'class'
            df[class_column_name] = y
            # map class label numerical value (stored as float) to class label string (fault class)
            df[class_column_name] = df[class_column_name].apply(lambda x: faults[int(x)])
            #print('df.index=', df.index)
            #print('df.columns=', df.columns)
            #print('Pandas data frame head=', df.head()) # ; quit()

            rad_viz = pd.plotting.radviz(df, class_column_name)
            plt.show()
            '''
            print('Generating Andrews curves plot...')
            rad_viz = pd.plotting.andrews_curves(df, class_column_name)
            plt.show()
            quit()
            '''

        else:
            print('Unknown visualization method (', method, ')')

        if method != 'RadViz':
            self.plot_condition(X, y, classlabel, classname, featname, plot_time_axis=plot_time_axis, time_offset=False,
                dropfigfile=dropfigfile, title=title)
        

    def datacsvreadTE(self, filename, delimiter = ' '):

        print('===> Reading TE data from file ', filename, '...')
        f = open(filename, 'rt')
        reader = csv.reader(f, delimiter=delimiter)
        row1 = next(reader)
        ncol = len(row1) # Read first line and count columns
        # count number of non-empty strings in first row
        nfeat = 0
        for j in range(ncol):
            cell = row1[j]
            if cell != '':
                nfeat = nfeat + 1
                #print('%.2e' % float(cell))

        f.seek(0)              # go back to beginning of file
        #print('ncol=', ncol, 'nfeat=', nfeat)
        
        x = np.zeros(1)
        X = np.empty((0, 1))
        r = 0
        for row in reader:
            #print(row)
            c = 0
            ncol = len(row)
            #modifiquei aqui
            for j in range(ncol):
                if j == ncol - 1:
                    cell = row[j]
                    if cell != '':
                        x[c] = float(cell)
                #print('r=%4d' % r, 'j=%4d' % j, 'c=%4d' % c, 'x=%.4e' % x[c])
                        c = c + 1
            r = r + 1
            X = np.append(X, [x], axis=0)
            #if r > 0: # DBG
            #    break
        #print('X.shape=\n', X.shape)#, '\nX=\n', X)
        return X

    def filter_vars(self, X, mask):
        return X[:,np.array(mask,dtype=int)], list(np.array(self.extendedfeatname)[mask])

    def signal_plot(self, infile=None, X=None, divide_by_mean=True, subtract_mean=False, standardize=False,
            dropfigfile=None, title=None, mask=None):

        if not infile is None:
            if X is None:
                print('===> Reading TE data from file ', infile, '...')
                X = self.datacsvreadTE(infile)
            else:
                print('Data X exist. Ignoring infile...')

        featname = self.extendedfeatname
        #print('featname=',featname,'mask=',mask)
        if not mask is None:
            '''
            mask = np.array(mask,dtype=int)
            X = X[:,mask]
            featname = list(np.array(extendedfeatname)[mask])
            '''
            X, featname = self.filter_vars(X, mask)

        n, d = X.shape
        #print(X)
        tsfig = plt.figure(2, figsize=(12,6)) # figsize in inches
        for j in range(d):
            ts = X.T[j,:]
            if divide_by_mean:
                ts = ts / np.mean(ts)
                ts = ts + j
            elif subtract_mean:
                ts -= np.mean(ts)
            elif standardize:
                ts -= np.mean(ts)
                ts /= np.var(ts)
            #print('Feat#', j+1, '=', ts)
            plt.plot(ts, linewidth=0.5)

        if not title is None:
            plt.title(title)
        # Legend ouside plot:
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        #plt.legend(featname, fontsize=7, loc='best')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', mode='expand')
        #plt.tight_layout(pad=70)
        #plt.legend(featname, fontsize=7, loc='best', bbox_to_anchor=(0.5, 0., 1.0, 0.5))
        plt.legend(featname, fontsize=7, loc='center left', bbox_to_anchor=(0.85, 0.60),
                fancybox=True, shadow=True, ncol=1)
        if not dropfigfile is None:
            print('Saving figure in ', dropfigfile)
            plt.savefig(dropfigfile, dpi=1200)
        plt.show()



def test1():
    import csv
    
    with open('all.csv') as csvfile:
        
        all_dados = list(csv.reader(csvfile, delimiter="\t"))
        all_dados = np.array(all_dados[1:], dtype=np.str)
        
        X = all_dados[:,0:all_dados.shape[1]-1]
        Y = all_dados[:,all_dados.shape[1]]
    return X, Y


def main():
    os.system("rm *.csv")
    #para gerar novos arquivos de treino
    os.system("bash commands.sh")
    print('Executing main() ....')
    make_all_csv()
    #X, Y = test1()
    #return        
    te = TE()
    n_components = 2
    ftest = 'fault_1.csv'
    X = te.datacsvreadTE(ftest) 
    n, d = X.shape
    y = np.zeros(n, dtype = int)   
    classlabel = np.array(range(100))
    
    ftest = 'fault_2.csv'
    Xi = te.datacsvreadTE(ftest)
    X = np.concatenate((X, Xi), axis = 0)
    n, d = Xi.shape
    yi = np.ones(n, dtype = int)
    y = np.concatenate((y, yi), axis = 0)
    featname = ['fault_1', 'fault_2']
    X_train = TSNE(n_components=n_components).fit_transform(X) #TSNE(n_components=n_components).fit_transform(X) X
    classname = ['fault_1', 'fault_2']
    classlabel = np.array([0, 1])
    featname = ['fault_1', 'fault_2']

    #print('ntrain=', ntrain, 'ntest=', ntest, 'n=', 'X=\n', X, '\nshape=', X.shape, 'y=\n',y,'shape=',y.shape,) #; quit()
    
    te.plot_condition(X_train, y, classlabel, classname, featname, plot_time_axis = True,
            dropfigfile = 'TSNE', title = 'TSNE')
        
    som = MiniSom(7, 7, 2, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X_train)
    print("Training...")
    num_iterations = 1000
    import matplotlib.pyplot as plt 
    import matplotlib.image as mgimg
    from matplotlib import animation
    for i in range(100):
        som.train_random(X_train, 1000) # training with 100 iterations
        print("Número de iterações %d!" %num_iterations)
        plt.bone()
        plt.clf()
        plt.pcolor(som.distance_map().T) # distance map as background
        plt.colorbar()
        markers = ['o','s','D']
        colors = ['r','g','b']
        for cnt,xx in enumerate(X_train):
            w = som.winner(xx) # getting the winner
            # palce a marker on the winning position for the sample xx
            plt.plot(w[0]+.5,w[1]+.5,markers[y[cnt]],markerfacecolor='None',
                markeredgecolor=colors[y[cnt]],markersize=12,markeredgewidth=2)
        plt.title('Mapa Auto-Organizável')
        plt.axis([0,som._weights.shape[0],0,som._weights.shape[1]])
        plt.savefig('som_' + str(num_iterations) + '.png') # show the figure
        num_iterations = num_iterations + 1000
        plt.clf()
    

    fig = plt.figure()
    myimages_0 = []
    num = 1000
    #loops through available png:s
    for p in range(100):

        ## Read in picture
        fname = "som_%d.png" %num 
        img = mgimg.imread(fname)
        imgplot = plt.imshow(img)

        myimages_0.append([imgplot])
        num = num + 1000
    my_anim = animation.ArtistAnimation(fig, myimages_0, interval=1000, blit=True, repeat_delay=1000)
    my_anim.save("som.gif") #plt.show()
    os.system("rm som_*.png")
   
if __name__ == "__main__":
    main()