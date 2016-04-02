import tensorflow as tf
import numpy as np
import math
import time

# column vectors are [n,1] matrix
# row vectors are [1,n] matrix
# in the comments, S(exp(i), i) is a sum over i of exp(i)
# and (exp)_i is a sequence indexed by i
# the comments first describe the tensor contents, and then its shape is given
def clusterize(batch, k=10, MAX_ITER=100, epsilon=10**-3):
    start = time.time()
    
    # k = number of clusters
    n = len(batch) #number of vectors
    p = len(batch[0]) # vectors dimension

    #initialisation
    points = tf.placeholder(tf.float32, shape=[n,p,1]) # x_i [n, p, 1]
    pi = tf.Variable(tf.random_uniform([1,k]))
    mu = tf.Variable(tf.slice(tf.random_shuffle(points), [0,0,0], [k,p,1]))
    seed = tf.random_uniform([k,p,p])
    sigma = tf.Variable(tf.batch_matmul(seed,tf.transpose(seed, [0,2,1]))) #[k,p,p]
    #we don't care for the initial value of thoses ones
    T = tf.Variable(tf.random_uniform([n,k]))  
    func = tf.Variable(tf.random_uniform([n,k]))

    coef = math.pow(2*math.pi, -p/2)
    def get_function(points, mu, sigma): # f_ik [n,k]
        div = coef*tf.rsqrt(tf.batch_matrix_determinant(sigma)) # ((2pi)^p*|S_k|)^-1/2  [k]
        div = tf.tile(tf.reshape(div, [1,k]), [n,1]) # [n,k]
        diff = tf.sub(tf.tile(points, [k,1,1]), tf.tile(mu, [n,1,1])) # x_i-u_k [n*k, p, 1]
        sigma = tf.tile(sigma, [n,1,1]) # [n*k,p,p]
        exp = tf.exp(-0.5*tf.batch_matmul( tf.transpose(diff,perm=[0,2,1]), tf.batch_matmul(tf.batch_matrix_inverse(sigma), diff) )) # e^(d'*S^-1*d)_ik [n*k, 1, 1]
        exp = tf.reshape(exp, [n,k])

        return tf.mul(div, exp) # Multivariate normal distribution evaluated for each vector, for each cluster parameter. Hence the [n,k] shape.
        
   
    def get_T(points, pi, func): #E step
        coefs = tf.mul(func, tf.tile(pi, [n, 1]))  # pi_k*f_ik [n,k] (actual probability density function in each point for each cluster. See final return value)
        sum = tf.reduce_sum(coefs, 1, keep_dims=True) # S(pi_l*f_il, l) [n,1]
        T = tf.div(coefs, tf.tile(sum, [1,k])) # t_ik [n,k]
        
        return T

    def get_pi_mu_sigma(points, T): #M step
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

    def get_Q(T, pi, func):
        return tf.reduce_sum(tf.mul(T,tf.log(tf.mul(tf.tile(pi, [n,1]), func))), [0,1])
        

    #actual operations that will be performed
    pi_tmp, mu_tmp, sigma_tmp = get_pi_mu_sigma(points, T)
    func_op = func.assign(get_function(points, mu, sigma))
    T_op = T.assign(get_T(points, pi, func))
    with tf.control_dependencies([func_op, T_op]):
        pi_op = pi.assign(pi_tmp)
        mu_op = mu.assign(mu_tmp)
        sigma_op = sigma.assign(sigma_tmp)


    converged = False
    iters = 0
    sess = tf.Session()
    sess.run(tf.initialize_all_variables(), feed_dict={points:batch})
    oldQ = sess.run(get_Q(T,pi,func), feed_dict={points:batch})
    while not converged and iters < MAX_ITER: #iterations loop
        iters +=1
        
        sess.run([func_op, T_op, pi_op, mu_op, sigma_op], feed_dict={points:batch})
        
        newQ = sess.run(get_Q(T,pi,func), feed_dict={points:batch})
        if abs((newQ - oldQ)/oldQ) < epsilon: #if Q doesn't change much 
            converged = True
        oldQ = newQ
    
    print("Converged in : ", time.time() - start)

    return sess.run(tf.mul(func, tf.tile(pi, [n, 1])), feed_dict={points:batch}) #the probability for each vector to belong to each cluster

if __name__ == "__main__":
    clusterize(tf.random_uniform([100,10,1]).eval(session=tf.Session()))
