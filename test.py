import keras
import hddm
import numpy as np

import scipy as sp
from scipy import stats
import numpy as np

class DDMSimulator(object):
    
    def __init__(self, n_trials=100):        
        self.a_dist = sp.stats.gamma(a=0.75*1.5, scale=1/1.5)
        self.v_dist = sp.stats.norm(2, 3)
        self.z_dist = sp.stats.norm(0, .5)
        
        self.sample_a()
        self.sample_v()
        self.sample_z()
        
        self.n_trials = n_trials
        
        self.train_array = np.zeros((self.n_trials, 5))
        
    def sample_a(self, n=None):
        self.a = self.a_dist.rvs(n)
    
    def sample_v(self, n=None):
        self.v = self.v_dist.rvs(n)    
    
    def sample_z(self, n=None):
        self.z = sp.special.expit(self.z_dist.rvs(n))
    
    def sample(self, n=None):
        self.sample_a(n)
        self.sample_v(n)
        self.sample_z(n)
    
    def get_random_rts(self, n=1000):
        self.sample()
        
        params = {'v':self.v,
                  'a':self.a,
                  'z':self.z,
                  't':0}
        
        hddm.utils.check_params_valid(**params)
        
        df = hddm.generate.gen_rts(size=n, v=self.v, a=self.a, z=self.z, t=0)
        
        hddm.utils.flip_errors(df)
        
        if not np.isfinite(df.rt).all():            
            raise Exception('nans in rt')
        
        return df.rt.values
    
    def get_train_data(self):
        
        rts = self.get_random_rts(self.n_trials)
        
        self.train_array[:, 0] = rts
        self.train_array[:, 1] = self.v
        self.train_array[:, 2] = self.a
        self.train_array[:, 3] = self.z
        
        ll = hddm.wfpt.pdf_array(x=rts, 
                                 v=self.v, 
                                 sv=0., 
                                 a=self.a, 
                                 z=self.z, 
                                 sz=0., 
                                 t=0.,
                                 st=0.,
                                 logp=True)
        
        
        self.train_array[:, 4] = ll
        
        if not np.isfinite(ll).all():
            return self.get_train_data()
                
        return self.train_array

        
        
        
        

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(128, input_shape=(4,), use_bias=True, kernel_initializer='normal'),
    Activation('relu'),
    Dense(128, use_bias=True),
    Activation('relu'),    
    Dense(128, use_bias=True),
    Activation('relu'),        
    Dense(128, use_bias=True),
    Activation('relu'),            
    Dense(1,  kernel_initializer='normal'),
])

model.compile(loss='mean_squared_error', optimizer='adam')


d = DDMSimulator()

for i in range(1000):
    print(i)
    data = np.concatenate([d.get_train_data() for i in range(100)], 0)
    print(model.train_on_batch(data[:, :4], data[:, 4:]))

        
