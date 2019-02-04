import numpy as np
def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    mat=np.zeros((N,L))
    back_pointers=np.zeros(emission_scores.shape,dtype=int)
    for i in xrange(L):
        mat[0][i]=start_scores[i]+emission_scores[0][i]
        back_pointers[0][i] = i
    y = []
    #score=0.0
    max_sum=[]
    for i in xrange(N-1):
        for l in xrange(L):
            max_prob=[]
            for l1 in xrange(L):
                max_prob.append(emission_scores[i+1][l]+trans_scores[l1][l]+mat[i][l1])
            mat[i+1][l]=max(max_prob)
            ind = np.argmax(max_prob)
            back_pointers[i+1][l] = ind
    #print(end_scores)
    #print(mat)
    #print(back_pointers)
    
    for i in xrange(L):
        max_sum.append(end_scores[i]+mat[N-1][i])
    score=max(max_sum)
    index=np.argmax(max_sum)
    y.append(index)    
    for i in xrange(N-1,0,-1):
        y.append(back_pointers[i][index])
        index = back_pointers[i][index]
    y.reverse()
    #print(y)
    return (score, y)
