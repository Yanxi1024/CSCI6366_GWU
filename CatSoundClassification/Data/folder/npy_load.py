import numpy

if __name__ == '__main__':
    filepath = './car_single.npy'
    a = numpy.load(filepath)
    print(a.shape)