# Script to generate new embeddings file by averaging over K-nearest neighbors
import numpy as np
import argparse


def write_to_file(str_to_print, filename):
    f = open(filename, 'a')
    f.write(str_to_print)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate new embeddings file by averaging over K-nearest neighbors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', nargs='?', type=str,
        default='../data/pft_sg1_s50_w4_m1_i10.npy',
        help='input embeddings .npy file')
    parser.add_argument(
        '-l', nargs='?', type=str,
        default='../data/output_knn_idlist.txt',
        help='output neighbors list file')
    parser.add_argument(
        '-o', nargs='?', type=str,
        default='../data/output_knn.npy',
        help='output embeddings file')
    parser.add_argument(
        '-t', nargs='?', type=str,
        default='../data/test.ann',
        help='index file name')
    parser.add_argument(
        '-k', nargs='?', type=int,
        default=20, help='number of neighbors')
    parser.add_argument(
        '-g', nargs='?', type=str,
        default='concat', help="mode of adding neighbors, options: 'avg' or 'concat'")
    args = parser.parse_args()
    
    arr = np.load(args.d)
    
    print('Loaded vectors shape:', arr.shape)
    
    size = arr.shape[0]
    dims = arr.shape[1]
        
    neighbor_lists = open(args.l).readlines()
    neighbor_lists = [eval(x.strip()) for x in neighbor_lists]
    if args.g == 'concat':
        dims = 2*dims
    new_arr = np.zeros(shape=(size, dims))
    for idx in range(size):
        idlist = neighbor_lists[idx][1:(args.k+1)]
        vec = np.average(arr[[i for i in idlist]], axis=0)
        if args.g == 'concat':
            new_arr[idx] = np.concatenate([arr[idx], vec], axis=0)
        else: 
            new_arr[idx] = np.average([arr[idx], vec], axis=0)
    np.save(args.o, new_arr)
