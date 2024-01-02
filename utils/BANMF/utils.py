import numpy as np
import sys

def divide(a, b): # c = a / b
    c = np.divide( a, b, out=sys.maxsize*np.ones(np.shape(a)), where=b!=0)
    return c
# Compute association matrix
def calculate_association(matrix, threshold=0.5):
    row, col = matrix.shape
    ASSO = np.zeros((col, col))
    for i in range(col):
        idx = (matrix[:, i] == 1)
        for j in range(col):
            ASSO[i, j] = sum(matrix[:, j][idx])
        if ASSO[i, i] != 0:
            ASSO[i, :] = ASSO[i, :] / ASSO[i, i]
    
    ASSO[ASSO >= threshold] = 1
    ASSO[ASSO < threshold] = 0

    return ASSO.astype(np.uint8)


def solve_basis(matrix, k, asso, bonus, penalty, binary=False):

    row, col = matrix.shape

    # Mark whether an entry is already covered
    covered = np.zeros((row, col)).astype(np.uint8)

    # Coefficient matrix for bonux or penalty
    coef = np.zeros(matrix.shape)
    coef[matrix == 0] = penalty
    coef[matrix == 1] = bonus

    # If in binary mode, make coef exponential
    if binary:
        coef *= np.array([2**e for e in range(col-1, -1, -1)])

    for i in range(k):

        best_basis = np.zeros((1, col)).astype(np.uint8)
        best_solver = np.zeros((row, 1)).astype(np.uint8)
        best_score = 0

        for b in range(col):
            # Candidate pair of basis and solver
            basis = asso[b, :]
            solver = np.zeros((row, 1)).astype(np.uint8)
            
            # Compute score for each row
            not_covered = 1 - covered
            score_matrix = coef * not_covered * basis
            score_per_row = np.sum(score_matrix, axis=1)

            # Compute solver
            solver[score_per_row > 0] = 1

            # Compute accumulate point
            score = np.sum(score_per_row[score_per_row > 0])

            if score > best_score:
                best_basis = basis
                best_solver = solver
                best_score = score
        
        # Stack matrix B and S
        if i == 0:
            B = best_basis.reshape((1, -1))
            S = best_solver.copy()
        else:
            B = np.vstack((B, best_basis))
            S = np.hstack((S, best_solver))
        
        # Update covered matrix
        covered = np.matmul(S, B)
        covered[covered > 1] = 1
    
    return S, B, covered





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
    maxW = np.max(W)
    minH = np.min(H[np.nonzero(H)])
    maxH = np.nanmax(H)
    maxH = np.max(H)
    # print(minW, maxW, minH, maxH)
    assert maxW == maxW
    assert maxH == maxH
    delta_W_candidiate = np.linspace(start=minW, stop=maxW, num=150, endpoint=True)
    delta_H_candidiate = np.linspace(start=minH, stop=maxH, num=150, endpoint=True)
    
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
            # norm = np.linalg.norm(X - W_hH_h, ord='fro') # calculate distance
            norm = weighted_HD(X, W_hH_h)
            if norm < min_distance:
                min_distance = norm
                best_W_head = W_head
                best_H_head = H_head
                best_delta_W = delta_w
                best_delta_H = delta_h
                
    return best_W_head, best_H_head, best_delta_W, best_delta_H, min_distance

def ASSO_algo (input_truth, k):
    row, col = input_truth.shape

    # Best pair
    best_B = -1
    best_S = -1
    best_result = -1
    best_score = float('inf')

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    ### ASSO algorithm ###
    for threshold in threshold_list:
        association = calculate_association(input_truth, threshold) # A
        S, B, result = solve_basis(input_truth, k, association, 1, -1, True)
        if True:
            score = weighted_HD(input_truth, result) # This is the new part proposed in BLASYS 3.2
        else:
            score = HD(input_truth, result)

        if score < best_score:
            best_B = B
            best_S = S
            best_result = result
            best_score = score
    
    ### Exhastive search on B ### 
    # Wait... why not search on S as ASSO proposed?
    # Enumerate possible columns
    column_list = []
    multi_list = []
    for i in range(2**k):
        binary_str = '{0:0>{1}}'.format(bin(i)[2:], k)
        column = np.array(list(binary_str)).astype(np.uint8)
        column_list.append(column)
        prod = np.matmul(best_S, column)
        prod = prod % 2
        multi_list.append(prod)
    
    # Brute force best column in B
    for i in range(col):
        ground_truth = input_truth[:, i]
        best_similar = 0
        best_idx = -1
        for j in range(2**k):
            similar = sum(multi_list[j] == ground_truth)
            if similar > best_similar:
                best_idx = j
                best_similar = similar
        best_B[:, i] = column_list[best_idx]
        
    return best_S, best_B


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
    if app.shape == 1:
        return 
    return np.sum(org != app)


def weighted_HD(org, app):
    assert org.shape == app.shape
    row, col = org.shape
    weight = np.array([2**e for e in range(col-1, -1, -1)])
    HD = (org != app)
    return np.sum(HD * weight)


