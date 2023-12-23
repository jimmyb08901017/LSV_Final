import numpy as np
import sys
import os
import shutil
from .utils import *

### 
# The part to change :
# ./utils/utils.py:348
### 
def BANMF(truthtable, k, binary = True, regularized=False):
    # Read in input truthtable
    input_truth = get_matrix(truthtable)
    row, col = input_truth.shape
    
    # [Note]
    # For BMF, we compute |C - S*B|
    # For BANMF, we compute |X - Y|, Y is made by W*H with threshold
    # B->H; S->W; 
    
    # Output path
    X_path = truthtable + '_x_' + str(k)
    write_matrix(input_truth, X_path)
    H_path = truthtable + '_h_' + str(k)
    W_path = truthtable + '_w_' + str(k)
    mult_path = truthtable + '_wh_' + str(k)
    # H_path = "BANMF" + '_h_' + str(k)
    # W_path = "BANMF" + '_w_' + str(k)
    # mult_path = "BANMF" + '_wh_' + str(k)
    
    # Initialize W, H, Y
    # [Note] Y, W, and H are not necessarily boolean matrix!    
    # We will convert W and H to boolean matrix in algo-2. 
    W = np.random.rand(row, k) # ! Entries in W & H are now [0, 1),
    H = np.random.rand(k, col) #   may be a problem. 
    Y = input_truth            ##  auxiliary matrix
    # print(">>> initial <<<")
    # print("W:")
    # print(W)
    # print("H:")
    # print(H)
    # print("Y:")
    # print(Y)

    
    ##  Several usages of numpy ## 
    # test1 = np.matmul(W, H) # mul
    # H.transpose() # transpose
    # (H/H) # element-wise calculation
    # H_threshold = np.where(H > 0.5, H, 0) # threshold
    
    
    # Number of iteration
    N_iter = 5
    
    ### Algorithm 1: BANMF algorithm ###
    if not regularized: 
        Y, W, H = BANMF_algo(k, input_truth, Y, W, H, N_iter)
    
    ### Algorithm 3: RegularizedBANMF algorithm ###
    if regularized:
        reg_lambda = 0.1 # for regularized BANMF
        Y, W, H = regularized_BANMF_algo(k, input_truth, Y, W, H, N_iter, reg_lambda)
        
    # print(">>> after <<<")
    # print("W:")
    # print(W)
    # print("H:")
    # print(H)
    # print("Y:")
    # print(Y)

    ## Algorithm 2: Booleanization ##
    best_W_head, best_H_head, \
    best_delta_W, best_delta_H, min_distance = booleanization(input_truth, W, H)
                
    # print("min_distance:", min_distance)
    # print(f"threshold w:{best_delta_W} h:{best_delta_H}")
    # print("result W:")
    # print(best_W_head)
    # print("result H:")
    # print(best_H_head)
    
    
    write_matrix(best_H_head, H_path)
    write_matrix(best_W_head, W_path)
    new_best_result = np.matmul(best_W_head, best_H_head)
    new_best_result = new_best_result % 2
    print("BANMF HD:",HD(input_truth, new_best_result ))
    write_matrix(new_best_result, mult_path)
    
# if __name__ == "__main__":
#     truthtable = np.ones((5,5), dtype=np.uint8)
#     BANMF(truthtable, 3, True)