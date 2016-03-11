import tensorflow as tf
import numpy as np
import math
import time

K = 5 #number of clusters

def clusterize(batch):
    start = time.time()
    
    n = len(batch) #number of vectors
    p = len(batch[0]) # vectors dimension

    points = tf.placeholder([n,p]) # x_i [n, 1, p]
    pi = tf.variable(tf.random_uniform([1,k]))
    mu = tf.variable(tf.random_uniform([k,1,p]))
    sigma = tf.variable(tf.random_uniform([k,p,p]))
    T = tf.variable(tf.random_uniform([n,k]))
    func = tf.variable(tf.random_uniform([n,k])

    two_Pi_pow_p = math.pow(2*math.pi, p)
    def get_function(points, mu, sigma):
        div = tf.rsqrt(two_Pi_pow_p*tf.batch_matrix_determinant(sigma)) # ((2pi)^p*|S_k|)^-1/2  [k]
        div = tf.tile(tf.reshape(diff, [1,K]), [n,k]) # [n,k]
        diff = tf.diff(tf.tile(points, [K,1]), tf.tile(mu, [n,1])) # x_i-u_k [n*k, 1, p]
        exp = tf.exp(-0.5*tf.batch_matmul(tf.transpose(diff,perm=[0,2,1]), tf.batch_matmul(tf.batch_matrix_inverse(sigma), diff))) # e^(dt*S^-1*d)_ik [n*k, 1, 1]
        exp = tf.reshape(exp, [n,K])
        return tf.mul(div, exp)
    
   
    def get_T_func(points, pi, mu, sigma): #E
        func = get_function(points, mu, sigma) # f_ik [n,k]
        coefs = tf.mul(func, tf.tile(pi, [n, 1]))  # pi_k*f_ik [n,k]
        sum = tf.reduce_sum(coefs, 1, keep_dims=True) # S(pi_l*f_il,l) [n,1]
        T = tf.div(coefs, tf.tile(sum, [1,k])) # t_ik [n,k]
        
        return T, func

    
    def get_pi_mu_sigma(T, points): #M
        Tk = tf.reduce_sum(T, 0, keep_dims=True) # S(t_ik, i) [1, k]
        pi = Tk/n  # pi_k [1,k]
        tmp = tf.tile(tf.reshape(points, [1,n,p]), [K,1,1])
        tmp2 = tf.tile(tf.reshape(T, [K,n,1]), [1,1,p]) #okay, I have no idea if this will work
        mu = tf.div(tf.reduce_sum(tf.mul(tmp,tmp2), 1, keep_dims=True), tf.tile(tf.reshape(Tk, [k,1,1]), [1,1,p]]))# mu_k [k,1,p]
        diff = tf.diff(tf.tile(points, [K,1]), tf.tile(mu, [n,1])) # x_i-u_k [n*k, 1, p]
        tmp_mat = tf.reshape(tf.batch_matmul(diff, tf.transpose(diff,perm=[0,2,1])), [k,n,p,p]) # (x_i-u_k)(x_i-u_k)t [k,n,p,p]
        sigma = tf.reduce_sum(tf.mul(tmp_mat, tf.tile(tf.reshape(T, [K,n,1,1]), [1,1,p,p])), 1) # S_k [k, p, p]
        
        return pi, mu, sigma

    
    def do_updates()

    def get_Q(T, pi, func):
        return tf.sum_reduce(tf.mul(T,tf.log(tf.mul(tf.tile(pi, [n,1]), func))), [0,1])


    converged = false
    oldQ = 
    while not converged and iters < MAX_ITER:
        iters++

        
        newQ = get_Q(T,pi,func)
        if abs((newQ - oldQ)/oldQ) < epsilon:
            converged = True
        else:
            oldQ = newQ
        
        
if __name__ == "__main__":
    clusterize(tf.random_uniform([1000,20], maxval=200, dtype=tf.int32).eval(session=tf.Session()))
