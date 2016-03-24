import tensorflow as tf
import numpy as np
import math
import time

MAX_ITER = 1000
k = 2 #number of clusters
epsilon = 10**-2

def clusterize(batch):
    start = time.time()
    
    n = len(batch) #number of vectors
    p = len(batch[0]) # vectors dimension

    points = tf.placeholder(tf.float32, shape=[n,p,1]) # x_i [n, p, 1]
    pi = tf.Variable(tf.random_uniform([1,k]))
    mu = tf.Variable(tf.random_uniform([k,p,1]))
    seed = tf.random_uniform([k,p,p])
    sigma = tf.Variable(tf.batch_matmul(seed,tf.transpose(seed, [0,2,1]))) #[k,p,p]
    T = tf.Variable(tf.random_uniform([n,k]))
    func = tf.Variable(tf.random_uniform([n,k]))

    two_Pi_pow_p = math.pow(2*math.pi, p)
    def get_function(points, mu, sigma): # f_ik [n,k]
        div_tmp = tf.rsqrt(two_Pi_pow_p*tf.batch_matrix_determinant(sigma)) # ((2pi)^p*|S_k|)^-1/2  [k]
        div = tf.tile(tf.reshape(div_tmp, [1,k]), [n,1]) # [n,k]
        diff = tf.sub(tf.tile(points, [k,1,1]), tf.tile(mu, [n,1,1])) # x_i-u_k [n*k, p, 1]
        sigma_tmp = tf.tile(sigma, [n,1,1]) # [n*k,p,p]
        exp = tf.exp(-0.5*tf.batch_matmul( tf.transpose(diff,perm=[0,2,1]), tf.batch_matmul(tf.batch_matrix_inverse(sigma_tmp), diff) )) # e^(dt*S^-1*d)_ik [n*k, 1, 1]
        exp = tf.reshape(exp, [n,k])
        return tf.mul(div, exp)
    
   
    def get_T(points, pi, func): #E
        coefs = tf.mul(func, tf.tile(pi, [n, 1]))  # pi_k*f_ik [n,k]
        sum = tf.reduce_sum(coefs, 1, keep_dims=True) # S(pi_l*f_il,l) [n,1]
        T = tf.div(coefs, tf.tile(sum, [1,k])) # t_ik [n,k]
        
        return T

    def get_pi_mu_sigma(points, T): #M
        Tk = tf.reduce_sum(T, 0, keep_dims=True) # S(t_ik, i) [1, k]
        pi = Tk/n  # pi_k [1,k]
        tmp = tf.tile(tf.reshape(T, [n,k,1,1]), [1,1,p,1]) # (T_ik)_p [n,k,p,1]
        tmp2 = tf.tile(tf.reshape(points, [n,1,p,1]), [1,k,1,1]) # (x_i)_k [n,k,p,1]
        mu = tf.div(tf.reduce_sum(tf.mul(tmp,tmp2), 0), tf.tile(tf.reshape(Tk, [k,1,1]), [1,p,1]))# mu_k [k,p,1]
        diff_tmp = tf.sub(tf.tile(points, [k,1,1]), tf.tile(mu, [n,1,1])) # x_i-u_k [n*k, p, 1]
        tmp_mat = tf.reshape(tf.batch_matmul( diff_tmp, tf.transpose(diff_tmp,perm=[0,2,1]) ), [n,k,p,p]) # (x_i-u_k)(x_i-u_k)t [k,n,p,p]
        sigma = tf.reduce_sum(tf.mul(tmp_mat, tf.tile(tf.reshape(T, [n,k,1,1]), [1,1,p,p])), 0) # S_k [k, p, p]
        sigma = tf.div(sigma, tf.tile(tf.reshape(Tk, [k,1,1]), [1,p,p]))
        return pi, mu, sigma

    pi_tmp, mu_tmp, sigma_tmp = get_pi_mu_sigma(points, T)

    func_op = func.assign(get_function(points, mu, sigma))
    T_op = T.assign(get_T(points, pi, func))
    with tf.control_dependencies([func_op, T_op]):
        pi_op = pi.assign(pi_tmp)
        mu_op = mu.assign(mu_tmp)
        sigma_op = sigma.assign(sigma_tmp)
    

    def get_Q(T, pi, func):
        return tf.reduce_sum(tf.mul(T,tf.log(tf.mul(tf.tile(pi, [n,1]), func))), [0,1])
        
    
    converged = False
    iters = 0
    sess = tf.Session()
    sess.run(tf.initialize_all_variables(), feed_dict={points:batch})
    oldQ = sess.run(get_Q(T,pi,func), feed_dict={points:batch})
    print(batch)
    while not converged and iters < MAX_ITER:
        iters +=1
        print(sess.run([func_op, T_op, pi_op, mu_op, sigma_op], feed_dict={points:batch}))
        #print(sess.run(
        print(oldQ)
        newQ = sess.run(get_Q(T,pi,func), feed_dict={points:batch})
        if abs((newQ - oldQ)/oldQ) < epsilon:
            converged = True
        else:
            oldQ = newQ
        
        


if __name__ == "__main__":
    clusterize(tf.random_uniform([10,2,1]).eval(session=tf.Session()))
