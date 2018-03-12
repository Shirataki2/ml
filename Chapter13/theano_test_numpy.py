import numpy as np
import theano
from theano import tensor as T

x = T.fmatrix(name='x')
x_sum = T.sum(x, axis=0)

calc_sum = theano.function(inputs=[x], outputs=x_sum)

ar = [[1, 2, 3, 4], [5, 6, 7, 8]]
print('Column Sum: ', calc_sum(ar))

ar = np.linspace(0.3, 3000, 10000, dtype=theano.config.floatX)
ar = ar.reshape(100, 100)
print('Column Sum: ', calc_sum(ar))
