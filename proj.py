import pandas as pd
import numpy as np
import cvxpy as cvx
from termcolor import cprint

pd.set_option('display.width', 120)
# %%

def cal_neg(w):
    l1 = np.linalg.norm(w, 1)
    if l1 != 0:
        return np.clip(-w, 0, None).sum() / l1
    else:
        return 0


def cal_gap(w, max_neg):
    return w.sum() - (1 - 2 * max_neg) * np.linalg.norm(w, 1)


def cvx_ref(v, max_neg):
    n = v.size
    w = cvx.Variable(n)
    prob = cvx.Problem(cvx.Minimize(cvx.norm2(w - v)),
                       [cvx.sum_entries(w) >= (1-2*max_neg) * cvx.norm1(w)])
    prob.solve()
    
    return np.asarray(w.value).ravel()


def eval_obj(w, w0, max_neg):
    d = np.linalg.norm(w - w0, 2)
    msg = 'dist = {:11.5g}, gap = {:11.5g}, neg = {:7.5f} / {:7.5f}'
    print msg.format(d, cal_gap(w, max_neg), cal_neg(w), max_neg)
    print 


def get_theta(w0, num_neg, num_zero, npr):
    sum_neg = -w0[: num_neg].sum()
    num_pos = w0.size - num_neg - num_zero
    sum_pos = w0[-num_pos:].sum()
    theta = (sum_neg - npr * sum_pos) / (num_neg + num_pos * npr ** 2) 
    return theta

def get_theta_fast(w0_cs, w0_cs_reverse, num_neg, num_zero, npr):
    sum_neg = 0 if num_neg == 0 else -w0_cs[num_neg-1]
    num_pos = w0_cs.size - num_neg - num_zero
    sum_pos = 0 if num_pos == 0 else w0_cs_reverse[-num_pos]
    theta = (sum_neg - npr * sum_pos) / (num_neg + num_pos * npr ** 2) 
    return theta

  
def get_thetas(w0, max_neg):
    npr = max_neg / (1 - max_neg) 
    w0_cs = w0.cumsum() 
    w0_cs_reverse = w0[::-1].cumsum()[::-1]
    
    # get all
    thetas = np.zeros((n, n))
    for nn in xrange(n):
        for nz in xrange(0, n - nn):
            thetas[nn, nz] = get_theta(w0, nn, nz, npr)
            if not np.isclose(get_theta(w0, nn, nz, npr), get_theta_fast(w0_cs, w0_cs_reverse, nn, nz, npr), atol=1e-9):
                print nn, nz, get_theta(w0, nn, nz, npr), get_theta_fast(w0_cs, w0_cs_reverse, nn, nz, npr)
                assert False
    return pd.DataFrame(thetas)


def get_sum_neg(w0_cs, num_neg):
    return -w0_cs[num_neg - 1]

def get_sum_pos(w0_rcs, num_pos):
    return  0 if num_pos == 0 else w0_rcs[-num_pos]


def get_ww(w0, num_neg, num_zero, theta, npr):
    ww = w0.copy()
    pos_start = num_neg + num_zero
    ww[: num_neg] += theta
    ww[num_neg: pos_start] = 0
    ww[pos_start: ] += theta * npr
    return ww


def get_dists(w0, max_neg):
    npr = max_neg / (1 - max_neg) 
    thetas = get_thetas(w0, max_neg)
    dists = np.zeros((n, n))
    for nn in xrange(n):
        for nz in xrange(0, n - nn): 
            theta = thetas.iloc[nn, nz]
            ww = get_ww(w0, nn, nz, theta, npr)
            gap = cal_gap(ww, max_neg)
            if gap <= -1e-6:
                coeff = np.inf
            else:
                coeff = 1
            dists[nn, nz] = np.linalg.norm(ww - w0) * coeff
    return pd.DataFrame(dists)


def get_gaps(w0, max_neg):
    npr = max_neg / (1 - max_neg) 
    thetas = get_thetas(w0, max_neg)
    gaps = np.zeros((n, n))
    for nn in xrange(n):
        for nz in xrange(0, n - nn): 
            theta = thetas.iloc[nn, nz]
            ww = get_ww(w0, nn, nz, theta, npr)
            gap = cal_gap(ww, max_neg)
            gaps[nn, nz] = gap
    return pd.DataFrame(gaps)


def get_ww_slow(w0, num_neg, num_zero, npr):
    ww = w0.copy()
    num_pos = w0.size - num_neg - num_zero
    pos_start = num_neg + num_zero
    sum_neg = -ww[: num_neg].sum()
    sum_pos = ww[pos_start:].sum()
    theta = (sum_neg - npr * sum_pos) / (num_neg + num_pos * npr ** 2) 
    ww[: num_neg] += theta
    ww[num_neg: pos_start] = 0
    ww[pos_start: ] += theta * npr
    cprint('{} {} {}, post start at {}'.format(num_neg, num_zero, ww.size - num_neg - num_zero, pos_start), 'red')
    print sum_neg, sum_pos, 'theta=', theta
    return ww


# %%    
max_neg = 0.2
#w0 = np.asarray([-0.4,  0.8])
#w0 = np.asarray([-0.4, -0.1,  0.8])  # z = 1
#w0 = np.asarray([-0.16, -0.14, -0.1,  0.8])  # z=0
#w0 = np.asarray([-0.16, -0.14, -0.1,  0.1, 0.7])  # z = 0
#w0 = np.asarray([-0.4, -0.12, 0.1,  0.8])  #z = 1
#w0 = np.asarray([-0.4, -0.12, -0.1,  0.8])  # z = 2
#w0 = np.asarray([-0.4, -0.12, -0.1,  0.3, 0.5])  # z = 2

#w0 = np.asarray([-0.4, -0.1, -0.01,  0.8])  # z = 1 and -0.01 is in pos group
w0 = np.asarray([-0.4, -0.1, -0.005, -0.005, 0.3, 0.5])
np.random.seed(1)
#w0 = np.random.randn(10) - 0.4
w0 = np.random.rand(10) - 2; w0[-1] = 1; w0[-2] = -0.01
w0[-3] = -0.02

w0_abs = np.abs(w0)
w0_argsort = np.argsort(w0) #[::-1]
w0 = w0[w0_argsort]
w0_abs = w0_abs[w0_argsort]

n = w0.size
npr = max_neg / (1 - max_neg)

wc = cvx_ref(w0, max_neg)

print 'thetas'
thetas = get_thetas(w0, max_neg)
print 'neg side offset'
print thetas
#print 'pos side offset'
#print thetas * npr
print 

dists = get_dists(w0, max_neg)
print dists
gaps = get_gaps(w0, max_neg)
print gaps
print 

# %%

# def fast_proj(w0, max_neg):
w0_cs = w0.cumsum()
w0_rcs = w0[::-1].cumsum()[::-1]

prev_theta = -np.inf
num_zero = 0
data = []
if np.sum(w0) >= (1-2 * max_neg) * np.linalg.norm(w0, 1):
    ww = w0.copy()
else: 
    for j in xrange(0, n):  # j is the index of the last negative
        num_neg = j + 1
        
        num_pos = n - num_neg - num_zero
        theta = get_theta_fast(w0_cs, w0_rcs, num_neg, num_zero, npr)
        data.append([num_neg, num_zero, num_pos, 
                     get_sum_neg(w0_cs, num_neg), get_sum_pos(w0_rcs, num_pos), theta])
        print num_neg, theta / (n - j - 1) , prev_theta / (n - j)
#        if theta / (n - j - 1) > prev_theta / (n - j):
        if theta < prev_theta:
            # roll back, num_neg is fixed 
            num_neg = num_neg - 1
            num_pos = n - num_neg
            theta = prev_theta
            
            if -theta < w0[j] < 0:  # start searching for zeros
                for k in xrange(j, n):
                    num_zero = k - num_neg + 1
                    num_pos = n - num_neg - num_zero
                    theta = get_theta_fast(w0_cs, w0_rcs, num_neg, num_zero, npr)
                    data.append([num_neg, num_zero, num_pos, 
                                 get_sum_neg(w0_cs, num_neg), get_sum_pos(w0_rcs, num_pos), theta])
                    
                    if theta > prev_theta:
                        num_zero -= 1
                        num_pos = n - num_neg - num_zero
                        theta = prev_theta
                        print 'break on first pos. num_zero = {}'.format(num_zero)
                        break
                    
                    if k == n - 1 or w0[k + 1] > 0: 
                        print 'break on first pos. num_zero = {}'.format(num_zero)
                        break
                    
                    prev_theta = theta
                    
            print 'break on num_neg =', num_neg
            break
        prev_theta = theta

    
    ddf = pd.DataFrame(data, columns=['num_neg', 'num_zero', 'num_pos', 'sum_neg', 'sum_pos', 'theta'])
    print
    print ddf    
    cprint('theta = {}'.format(theta), 'blue') 
    
    ww = w0.copy()
    pos_start = num_neg + num_zero
    ww[: num_neg] += theta
    ww[num_neg: pos_start] = 0
    ww[pos_start: ] += theta * npr


print 'init'
eval_obj(w0, w0, max_neg)
print 'cvx'
eval_obj(wc, w0, max_neg)
print 'fast results'
eval_obj(ww, w0, max_neg)


wdf = pd.DataFrame({'w0': w0, 'wc': wc})
wdf['dc'] = wdf.wc - wdf.w0
wdf['rc'] = wdf.dc.max() / wdf.dc
wdf['ww'] = ww
wdf['dw'] = wdf.ww - wdf.w0
wdf['rw'] = wdf.dw.max() / wdf.dw
print wdf
print 'corr =', wdf.wc.corr(wdf.ww)

theta0 = wdf['dc'].loc[0]
cprint('theta = {}'.format(theta0), 'green')
