import numpy as np
import os
import shutil
from .utils import *

def BANMF(truthtable, k, binary = False)
    # Read in input truthtable
    input_truth = get_matrix(truthtable)
    row, col = input_truth.shape
    
    # [Note]
    # For BMF, we compute |C - S*B|
    # For BANMF, we compute |X - Y|, Y is made by W*H with threshold
    # B->H; S->W; 
    
    # Output path
    H_path = truthtable + '_h_' + str(k)
    W_path = truthtable + '_w_' + str(k)
    mult_path = truthtable + '_wh_' + str(k)
    
    # Todo: Random W & H
    W = np.random.rand(row, k) # ! Entries in W & H are now [0, 1), 
    H = np.random.rand(k, col) #   may be a problem. 
    Y = input_truth
    
    # Number of iteration
    N_iter = 100
    
    ### BANMF algorithm ###
    for i in range(N_iter):
        # Todo: Update W
        # Todo: Update H
        # Todo: Update Y
        pass
    
    ### RegularizedBANMF algorithm ###
    reg_lambda = 0.01 # for regularized BANMF
    for i in range(N_iter):
        # Todo: Update W
        # Todo: Update H
        # Todo: Update Y
        pass
    
    

    
    write_matrix(best_H, H_path)
    write_matrix(best_W, W_path)
    new_best_result = np.matmul(best_W, best_H)
    new_best_result = new_best_result % 2
    write_matrix(new_best_result, mult_path)