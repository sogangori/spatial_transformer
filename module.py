import numpy as np
import tensorflow as tf


def affine_grid_generator(height, width, theta):
    num_batch = tf.shape(theta)[0]
    x = tf.lin_space(-1.0, 1.0, width)
    y = tf.lin_space(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])#(3, h*w)
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    #sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))#(num_batch, 3, h*w)
    sampling_grid = tf.tile(sampling_grid, [num_batch, 1, 1])#(num_batch, 3, h*w)
    theta = tf.cast(theta, tf.float32)
    sampling_grid = tf.cast(sampling_grid, tf.float32)
        
    batch_grids = tf.matmul(theta, sampling_grid)#(m, 2, 3)@(m, 3, h*w)=(m,2,h*w)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])
    return batch_grids        

def get_pixel_value(img, x, y):
    #img (m,h,w,c)
    #x,y (m,h,w)
    shape = tf.shape(x)
    m = shape[0]
    h = shape[1]
    w = shape[2]
    batch_idx = tf.range(0, m)
    batch_idx = tf.reshape(batch_idx, [m,1,1])
    b = tf.tile(batch_idx, [1, h, w])
    
    indices = tf.stack([b, y, x], axis=3) #(m,h,w,3)
    
    return tf.gather_nd(img, indices)

def bilinear_sampler(img, batch_grids):
    #batch_grids (m, 2, h, w)
    #img (m,h,w,c)
    uv_x = batch_grids[:, 0]
    uv_y = batch_grids[:, 1]    
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H-1, tf.float32)
    max_x = tf.cast(W-1, tf.float32)
    #x [-1, 1]
    x = 0.5 * ((uv_x + 1.0) * max_x)
    y = 0.5 * ((uv_y + 1.0) * max_y)
    
    #grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.floor(x)#precision bad?
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1
    
    #clip out of boundary index
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
        
    #deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    wa = tf.expand_dims(wa, -1)
    wb = tf.expand_dims(wb, -1)
    wc = tf.expand_dims(wc, -1)
    wd = tf.expand_dims(wd, -1)
    
    x0 = tf.cast(x0, tf.int32)
    x1 = tf.cast(x1, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    y1 = tf.cast(y1, tf.int32)
    
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)
    
    #out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])    
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    
    return out

def spatial_transformer_network(net, theta):
    '''
    return out
    '''
    h = tf.shape(net)[1]
    w = tf.shape(net)[2]
    batch_grids = affine_grid_generator(h,w,theta)
    out = bilinear_sampler(net, batch_grids)
    return out