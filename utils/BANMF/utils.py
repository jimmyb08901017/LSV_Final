import numpy as np
import sys

def divide(a, b): # c = a / b
    c = np.divide( a, b, out=sys.maxsize*np.ones(np.shape(a)), where=b!=0)
    return c

### Algorithm 1: BANMF algorithm ###
def BANMF_algo(k, X, Y, W, H, N_iter, N_sample):
    for round in range(N_sample):
        best_W = np.ones(1)
        best_Y = np.ones(1)
        best_H = np.ones(1)
        best_dist = 100000000    
        for i in range(N_iter):
            # print(f">>> iteration {i} <<<")
            
            # Update W
            WH = np.matmul(W, H)
            H_t = H.transpose()
            W = W * np.matmul(Y, H_t)
            divider = np.matmul(WH, H_t)
            W = divide(W, divider)
            # W_new = W_new / np.matmul(WH, H_t)
            
            # Update H
            WH = np.matmul(W, H)
            W_t = W.transpose()
            H = H * np.matmul(W_t, Y) 
            divider = np.matmul(W_t, WH)
            H = divide(H, divider)
            #H_new = H_new / np.matmul(W_t, WH)
            
            # Update Y
            WH = np.matmul(W, H)
            Y = np.where( WH > 1 , WH, 1)
            Y = np.where( Y > k , k, Y)
            Y = np.where( X == 0, 0, Y)
            # Y = np.where( input_truth == 0, Y, 0)
            # W = W_new
            # H = H_new
            
            # dist = np.linalg.norm(Y - WH, ord='fro')
            dist = weighted_HD(Y, WH)
            # print("dist:", dist)
            if dist == 0:
                break
            if dist < best_dist:
                best_Y = Y
                best_W = W
                best_H = H
    return best_Y, best_W, best_H

### Algorithm 3: RegularizedBANMF algorithm ###
def regularized_BANMF_algo(k, X, Y, W, H, N_iter, N_sample, reg_lambda):
    for round in range(N_sample):
        
        best_W = np.ones(1)
        best_Y = np.ones(1)
        best_H = np.ones(1)
        best_dist = 100000000
        for i in range(N_iter):
            # print(f">>> iteration {i} <<<")
            
            # Update W
            WH = np.matmul(W, H)
            H_t = H.transpose()
            W2 = W*W
            W3 = W2*W
            W = W * (np.matmul(Y, H_t) + 3*reg_lambda*W2)
            divider = np.matmul(WH, H_t) + 2*reg_lambda*W3 + reg_lambda*W2
            W = divide(W, divider)
            
            # Update H
            H2 = H*H
            H3 = H2*H
            WH = np.matmul(W, H)
            W_t = W.transpose()
            H = H * (np.matmul(W_t, Y) + 3*reg_lambda*H2)
            divider = np.matmul(W_t, WH) + 2*reg_lambda*H3 + reg_lambda*H2
            H = divide(H, divider)
            
            
            # Update Y
            WH = np.matmul(W, H)
            Y = np.where( WH > 1 , WH, 1)
            Y = np.where(Y > k , k, Y)
            Y = np.where( X == 0, 0, Y)
            
            # print("W:")
            # print(W)
            # print("H:")
            # print(H)
            # print("WH:")
            # print(WH)
            # print("Y:")
            # print(Y)
            #dist = np.linalg.norm(Y - WH, ord='fro')
            dist = weighted_HD(Y, WH)
            # print("dist:", dist)
            if dist == 0:
                break
        if dist < best_dist:
            best_Y = Y
            best_W = W
            best_H = H
    return best_Y, best_W, best_H
        
## Algorithm 2: Booleanization ##
def booleanization(X, W, H):
    minW = np.min(W[np.nonzero(W)])
    maxW = np.nanmax(W)
    maxW = np.max(W[W<100])
    minH = np.min(H[np.nonzero(H)])
    maxH = np.nanmax(H)
    maxH = np.max(H[H<100])
    # print(minW, maxW, minH, maxH)
    assert maxW == maxW
    assert maxH == maxH
    delta_W_candidiate = np.linspace(start=minW, stop=maxW, num=300, endpoint=True)
    delta_H_candidiate = np.linspace(start=minH, stop=maxH, num=300, endpoint=True)
    
    best_W_head = []
    best_H_head = []
    best_delta_W = 0
    best_delta_H = 0
    min_distance = 10000
    
    for delta_w in delta_W_candidiate: # grid search
        W_head = np.where(W >= delta_w , 1, 0)
        # W_head = np.where(W_head == np.nan, 1, 0)
        for delta_h in delta_H_candidiate:
            H_head = np.where(H >= delta_h, 1, 0)
            # H_head = np.where(H_head == np.nan, 1, 0)
            
            
            # calculate | X - W^H^|
            W_hH_h = np.matmul(W_head, H_head) % 2
            norm = np.linalg.norm(X - W_hH_h, ord='fro') # calculate distance
            # norm = weighted_HD(X, W_hH_h)
            if norm < min_distance:
                min_distance = norm
                best_W_head = W_head
                best_H_head = H_head
                best_delta_W = delta_w
                best_delta_H = delta_h
                
    return best_W_head, best_H_head, best_delta_W, best_delta_H, min_distance


def get_matrix(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    mat = [list(i.strip().replace(' ', '')) for i in lines]
    return np.array(mat, dtype=np.uint8)


def write_matrix(mat, file_path):
    with open(file_path, 'w') as f:
        for row in mat:
            for ele in row:
                f.write('{} '.format(ele))
            f.write('\n')


def HD(org, app):
    assert org.shape == app.shape
    return np.sum(org != app)


def weighted_HD(org, app):
    assert org.shape == app.shape
    row, col = org.shape
    weight = np.array([2**e for e in range(col-1, -1, -1)])
    HD = (org != app)
    return np.sum(HD * weight)


