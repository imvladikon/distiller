import numpy as np

def match_sorted_array_to_another_sorted_array(I, J, matching_function=None):
    if matching_function is None:
        def match(I, i, J, j):
            return I[i] >= J[j]
        matching_function = match
    result = np.empty(len(I))
    i = 0
    j = 0
    N_I = len(I)
    N_J = len(J)
    while True:
        if j>=N_J:
            break
        if i>=N_I:
            break
        match = matching_function(I, i, J, j)
        if match>0:
            j += 1
        elif match<0:
            i += 1
        else:
            result[i] = j-1

    return result


if __name__=='__main__':
    I = [1,1,1,1,2,2.1,2.2,2.7,3,3.6,7,7.9,12]
    J = [0,1,2,3,4,6,7,8,9]

    result = match_sorted_array_to_another_sorted_array(I, J)
    print(result)