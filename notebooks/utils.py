import json, gzip
import numpy as np

# 20 standard amino acids
aa2idx = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9,
          'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}

############################################################
# load 'phipsi10882' dataset
############################################################
def load_phipsi():
    # read .json file
    with gzip.open('../data/phipsi.json.gz', 'rb') as f:
        dataset = json.load(f)

    # reduse dataset to a list for simpler access
    dataset = dataset['phipsi10882']

    # convert data to numpy arrays skipping first and last residues
    for item in dataset:
        n = len(item['sequence'])
        item['sequence'] = np.array([aa2idx[aa] for aa in item['sequence'][1:n-1]], dtype=np.int8)
        item['phi'] = np.array(item['phi'], dtype=np.float32)[1:n-1]
        item['psi'] = np.array(item['psi'], dtype=np.float32)[1:n-1]

        # convert (phi,psi) to their sin() and cos()
        # (4 numbers per angle pair)
        item['avec'] = np.vstack([
            np.sin(item['phi']).T,
            np.cos(item['phi']).T,
            np.sin(item['psi']).T,
            np.cos(item['psi']).T ]).T

    return dataset


############################################################
# given WINDOW, extract feature vector X from the dataset
############################################################
def getX(dataset, WINDOW):
    
    for item in dataset:
        seq = item['sequence']
        l = len(seq)
        # split current sequence in all possible
        # chunks of length WINDOW
        chunks = np.vstack([w for shift in range(0,WINDOW,1) 
                            for w in np.split(seq[shift:],range(0,l,WINDOW)) 
                            if len(w) == WINDOW])
        # 1-hot encode these chunks
        # and save in a temp. vector
        item['X'] = np.array(np.eye(20)[chunks], dtype=np.int8).reshape((chunks.shape[0],-1))
    
    # stack X vectors from all proteins together
    X = np.vstack([item['X'] for item in dataset])

    # delete temp. vectors
    for item in dataset:
        del item['X']
    
    return X


############################################################
# given WINDOW and clustering object KMEANS,
# extract label vector Y from the dataset
############################################################
def getY(dataset, WINDOW, KMEANS):

    for item in dataset:
        l = len(item['sequence'])
        # assign angle vectors to clusters
        abin = np.array(KMEANS.predict(item['avec']), dtype=np.int8)
        # for every WINDOW position, pick the element in the middle and
        # save corresponding cluster ID in temp. item['Y']
        item['Y'] = np.hstack([w[WINDOW//2] 
                               for shift in range(0,WINDOW,1) 
                               for w in np.split(abin[shift:],range(0,l,WINDOW)) 
                               if len(w) == WINDOW])
    # stack Y vectors from all proteins together
    Y = np.hstack([item['Y'] for item in dataset])
    
    # delete temp. vectors
    for item in dataset:
        del item['Y']
    
    return Y

############################################################
# given WINDOW, extract phi
############################################################
def getPHI(dataset, WINDOW):
    
    for item in dataset:
        phi = item['phi']
        l = len(phi)
        item['phi_temp'] = np.hstack([w[WINDOW//2] 
                                      for shift in range(0,WINDOW,1) 
                                      for w in np.split(phi[shift:],range(0,l,WINDOW)) 
                                      if len(w) == WINDOW])

    PHI = np.hstack([item['phi_temp'] for item in dataset])

    for item in dataset:
        del item['phi_temp']
    
    return PHI


############################################################
# given WINDOW, extract psi
############################################################
def getPSI(dataset, WINDOW):
    
    for item in dataset:
        psi = item['psi']
        l = len(psi)
        item['psi_temp'] = np.hstack([w[WINDOW//2] 
                                      for shift in range(0,WINDOW,1) 
                                      for w in np.split(psi[shift:],range(0,l,WINDOW)) 
                                      if len(w) == WINDOW])

    PSI = np.hstack([item['psi_temp'] for item in dataset])

    for item in dataset:
        del item['psi_temp']
    
    return PSI


############################################################
# root-mean squared error between two sets of angles
############################################################
def ang_rmse(ref, pred):
    # don't forget about periodicity!
    rmse = np.sqrt(np.sum(np.square(np.minimum(
        np.abs(pred - ref), 
        np.abs(pred - ref + 2*np.pi),
        np.abs(pred - ref - 2*np.pi))))
                   / ref.shape[0])
    return rmse
