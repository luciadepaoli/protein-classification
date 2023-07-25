import numpy as np
from Bio.PDB import PDBList
from pandas import DataFrame
import requests, json
import random
import os 
from biopandas.pdb import PandasPdb
import sys
from six.moves import urllib
from numba import jit,float32
from pathlib import Path
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import urllib
import csv as cs
import shutil as sht
import glob as gbl
import warnings
from IPython.display import display, Markdown
import py3Dmol
warnings.filterwarnings("ignore")

import sklearn.manifold as skm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from astropy.stats import sigma_clip
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from astropy.stats import sigma_clip
import warnings
warnings.filterwarnings('ignore')

def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):    
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except:
        return None
    
# Extract the x,y,z coordinates of the CA atoms for a given protein

def get_coord(x):

    try:
        ppdb = PandasPdb().fetch_pdb(x.lower())
        mainchain = ppdb.df['ATOM'][(ppdb.df['ATOM']['atom_name'] == 'CA')]

        a = mainchain[['x_coord', 'y_coord', 'z_coord']]

        N = len(a)
        coord = np.zeros((N,3))
        coord = np.array(a)

        return coord
    except AttributeError:
        return int(0)
    
# Reject a protein if the number of missing CA atoms is >3

def missing_atoms (inFile,h=3):

    """
    Input inFile-> PDB File, h-> How much missing CA you allowed?
    """
    c = 0
    FLAG = False
    # If j=TRUE the protein is ok
    j = True
    missing_atoms = [] 
    k = False
    for line in inFile:
        if FLAG==True and line.split()[1]=='470':
            k = True
            line[28:].replace('  ','   ')
            atoms = line[28:].split('   ')
            c += np.sum(np.char.count(atoms,"CA"))
        if ['REMARK','470'] != line.split(' ')[0:2]:
            FLAG = False
        if 'REMARK 470   M RES CSSEQI  ATOMS' == line.strip():
            FLAG = True
            atoms = []
        if c>h:
            j = False
            break
        
    return j

# Get coord of a protein and save it in a txt file inside /prot_coord

def get_data(N):

    unique_numbers = np.sort(random.sample(range(0, len(df_new["CATH_ID"])),N+int(0.2*N)))
    c = 0
    cath_domain = []
    prot_name = []
    pdbl = PDBList()
    if not os.path.exists("./prot_coord/"):
        os.makedirs("./prot_coord/")
    np.sort(df_new["CATH_ID"])
    for i in unique_numbers:
        prot_name.append((df_new["CATH_ID"][i]).lower())
        j = df_new["CATH_ID"][i]

        download_pdb(j, 'PDB')
        flag = missing_atoms("PDB/" + j + ".pdb")
 
        if os.path.exists('./PDB/' + j + '.pdb') == True:
            os.remove("PDB/" + j + ".pdb")  
            
        if flag == True:
            coord = get_coord(j)
            if type(coord)!= int:
                np.savetxt("./prot_coord/" + j + '.txt', coord)
                c += 1
                a=f"{df_new['Class_Number'][i]}.{df_new['Architecture_number'][i]}.{df_new['Topology_number'][i]}.{df_new['Homologous_superfamily_number'][i]}"
                cath_domain.append(a)
        if (c == N):
            break
        

    return cath_domain

@jit(nopython=True)
def tripleprod(a,b,c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0])

@jit(nopython=True)
def dist(a):
    return np.sqrt(a[1]**2 + a[0]**2 + a[2]**2)

@jit(nopython=True)
def writhe_matrix(M):

    N    = M.shape[0]
    W    = np.zeros((N-1,N-1))

    for i in range(N-1):
        dri  = M[i+1,:] - M[i,:]
        drih = 0.5*(M[i+1,:] + M[i,:])
        
        for j in range(i+1,N-1):
            drj = M[j+1,:] - M[j,:]
            drij = 0.5*(M[j+1,:] + M[j,:]) - drih
            
            v = tripleprod(dri,drij,drj)
            if(v!=0):
                k = dist(drij)
                k = k**3
                W[i,j] = v/k
    return W

@jit(nopython=True)
def create_I2(prot_coord):
    W = writhe_matrix(prot_coord)

    v = np.zeros(15)
    N = len(W)
    # Length of the protein
    v[0] = N+1

    for i in range(N-1):
        for j in range(i+2,N-1):
            v[1] += W[i,j]
            v[2] += abs(W[i,j])
            v[3] += W[i,i+1]*W[j,j+1]
            v[4] += abs(W[i,i+1])*W[j,j+1]
            v[5] += W[i,i+1]*abs(W[j,j+1])
            v[6] += abs(W[i,i+1]*W[j,j+1])
    for i in range(N-2):
        for j in range(i+3,N-2):

            v[7] += W[i,i+2]*W[j,j+2]
            v[8] += abs(W[i,i+2])*W[j,j+2]
            v[9] += W[i,i+2]*abs(W[j,j+2])
            v[10] += abs(W[i,i+2]*W[j,j+2])
    for i in range(N-3):
        for j in range(i+1,N-1):
            v[11] += W[i,i+3]*W[j,j+1]
            v[12] += abs(W[i,i+3])*W[j,j+1]
            v[13] += W[i,i+3]*abs(W[j,j+1]) 
            v[14] += abs(W[i,i+3]*W[j,j+1])
    v[1:2] /=v[0]   
    v[3:N-1] /= v[0]**2
    return v