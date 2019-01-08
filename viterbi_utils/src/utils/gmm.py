'''
Created on Jun 1, 2018

@author: kuehne
'''
import numpy as np

class GMM(object):
    '''
    classdocs
    '''


    def __init__(self, mean_file, var_file, hmm_state_file, dict_file):
        self.mean_file = mean_file;
        f = open(mean_file, 'r')
        self.mean = np.genfromtxt(f, delimiter=' ')
        f.close()
        self.var_file = var_file;
        f = open(var_file, 'r')
        self.var = np.genfromtxt(f, delimiter=' ')
        f.close()
        self.hmm_dict = [];
        with open(dict_file, 'r') as f:
            content = f.read().split('\n')[0:-1];
            line_count = 0;
            for line in content:
                self.hmm_dict.append(line.strip());
                line_count = line_count + 1;
                
        self.hmm_state_count = [];
        with open(hmm_state_file, 'r') as f:
            content = f.read().split('\n')[0:-1];
            line_count = 0;
            for line in content:
                self.hmm_state_count.append( int(line.strip()) );
                line_count = line_count + 1;
    
    def getObsProbs(self, dataset):
        
        # read the data
        data = dataset.features.values()[0] ;
        data = np.transpose(data);
        
        log_probs = np.zeros( (dataset.n_frames, dataset.n_classes), dtype=np.float32 )

        gConst = np.zeros(len(self.var));
        count_el = 0;
        for var_el in self.var :
            
            n = len(var_el)
            sum_gConst = n* np.log(2*np.pi);
            for iv in var_el :
                sum_gConst = sum_gConst + np.log(iv);
            
                
            gConst[count_el] =  sum_gConst ;
            count_el = count_el + 1;
        
        # compute porbs for each observation
        n_dims = data.shape;
        
        count_data = 0; 
        
        for data_el in data:
            
            count_state = 0; 
            
            for var_el in self.var :

                x = data_el;

                sum_ = gConst[count_state];
                count_el = 0;
                
                for i_el in data_el :
                    xmm = i_el - self.mean[count_state, count_el];
                    sum_ = sum_ + (xmm * xmm * (1/ self.var[count_state, count_el]) );
                    count_el = count_el + 1;

                sum_ = -0.5 * sum_;

                log_probs[count_data, count_state] = sum_;
                count_state = count_state + 1;
            
            count_data = count_data+1;         
            
        return log_probs

        