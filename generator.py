import os
import math
import itertools
import numpy as np

class GeneratorAE:

    def __init__(self,*, validation_percent=0.2, test_percent=0.1, batch_size=32):
        self.validation_percent = validation_percent
        self.test_percent = test_percent
        self.batch_size = batch_size
        
    def get_generators(self, orig_path=os.getcwd()+'/transformed_data/orig/',
                       noised_path=os.getcwd()+'/transformed_data/noised/'):
        self.orig_path = orig_path
        self.noised_path = noised_path
        self.file_names = self._refine_size_to_batch_size_(self._get_file_name_pairs_(self.orig_path,
                                                                                      self.noised_path))
        num_of_batches = int(len(self.file_names) / self.batch_size)
        num_of_test_epochs = int(num_of_batches * self.test_percent)
        num_of_validation_epochs = int(num_of_batches *
                                       self.validation_percent)
        return (self._generator_([0, num_of_test_epochs]), num_of_test_epochs,
                self._generator_([num_of_test_epochs, num_of_test_epochs + num_of_validation_epochs]), num_of_validation_epochs,
                self._generator_([num_of_test_epochs + num_of_validation_epochs, num_of_batches]), num_of_batches - num_of_test_epochs - num_of_validation_epochs)
        
    
    def _get_file_name_pairs_(self, orig_path=os.getcwd()+'/transformed_data/orig/',
                              noised_path=os.getcwd()+'/transformed_data/noised/'):
        matching_files = []
        orig_file_names = os.listdir(orig_path)
        noised_file_names = os.listdir(noised_path)

        for curr_orig in orig_file_names:
            if curr_orig in noised_file_names:
                matching_files.append(curr_orig)
        return matching_files
    
    def _refine_size_to_batch_size_(self, file_names):
        return file_names[0:(math.floor(len(file_names)/self.batch_size))*self.batch_size]
    
    def _generator_(self, intervall):
        start_indexes = itertools.cycle(list(range(intervall[0]*self.batch_size,
                                                   intervall[1]*self.batch_size,
                                                   self.batch_size)))
        while True:
           curr_start_index = next(start_indexes)
           curr_list_orig = []
           curr_list_noised = []
           for curr_index in range(curr_start_index, curr_start_index+self.batch_size):
               curr_list_orig.append(np.load(self.orig_path + self.file_names[curr_index]))
               curr_list_noised.append(np.load(self.noised_path + self.file_names[curr_index]))
           yield (np.asarray(curr_list_noised).reshape((self.batch_size, 100, 93680, 1)),
                  np.asarray(curr_list_orig).reshape((self.batch_size, 100, 93680, 1)))

    