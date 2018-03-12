import theano
from theano import tensor as T

x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1*x1+w0

net_input = theano.function(inputs=[x1, w1, w0], outputs=z1)

print('Net Input: %.2f' % net_input(5.0, 3.0, -4.0))
