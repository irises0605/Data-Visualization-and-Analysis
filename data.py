'''data.py
Reads CSV files, stores data, access/filter data by variable name
Iris Liu
CS 251 Data Analysis and Visualization
Spring 2020
'''

import csv
import numpy as np

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor
        '''
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = {}
        self.types = []
        
        if self.filepath != None:
            self.read(self.filepath)

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data`
        '''
        rows = []
        with open(filepath, 'r') as csvfile:
            self.data = csv.reader(csvfile, quoting=csv.QUOTE_NONE, quotechar = '|', escapechar='\\')
            self.headers = next(self.data)
            self.types = next(self.data)
            
            for row in self.data:
                rows.append(row)
        
            temp = self.types
            charFieldIdx = []
            for i in range(len(temp)):
                temp[i] = temp[i].strip()
                self.headers[i] = self.headers[i].strip()
                if temp[i] != 'numeric':
                    charFieldIdx.append(int(i))
                if len(charFieldIdx) == len(self.types):
                    print("*Data not readable*")
                    print("*Data file needs to follow the standard*")
                    print("*Your file does not include data types or none of the types is nummeric*")
                    return
            
            j = 0
            for i in charFieldIdx:
                self.headers.pop(i-j)
                self.types.pop(i-j)
                j = j + 1
        
            for row in rows:
                j = 0
                for i in charFieldIdx:
                    row.pop(i-j)
                    j = j + 1
            
            for i in range(len(rows)):
                temp = []
                for field in rows[i]:
                    field = float(field)
                    temp.append(field)
                rows[i] = temp
            
            self.data = rows
            self.data = np.array(self.data)
        
        dict = {}
        for i in range(self.get_num_dims()):
            dict[self.headers[i]] = i
        self.header2col = dict
        
    def __str__(self):
        '''toString method
        '''
        string = ""
        for row in self.data[:5]:
        	for col in row:
        		string = string + "%10s"%col
        	string = string + '\n' 
        
        return string

    def get_headers(self):
        '''Get method for headers
        '''
        return self.headers

    def get_types(self):
        '''Get method for data types of variables
        '''
        return self.types

    def get_mappings(self):
        '''Get method for mapping between variable name and column index
        '''
        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample
        '''
        return len(self.headers)

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset
        '''
        return len(self.data)

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)
        '''
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.
        '''
        indices = []
        for i in range(len(headers)):
            for j in range(len(self.headers)):
                if headers[i] == self.headers[j] :
                    indices.append(j)
        return indices

    def get_all_data(self):
        '''Gets a copy of the entire dataset
        '''
        copy = self.data.copy()
        return copy

    def head(self):
        '''Return the 1st five data samples (all variables)
        '''
        if self.get_num_samples() <= 5:
            return self.get_all_data()
        else:
            return np.array([self.data[0],
                            self.data[1],
                            self.data[2],
                            self.data[3],
                            self.data[4]])

    def tail(self):
        '''Return the last five data samples (all variables)
        '''
        if self.get_num_samples() <= 5:
            return self.get_all_data()
        else:
            return np.array([self.data[-5],
                            self.data[-4],
                            self.data[-3],
                            self.data[-2],
                            self.data[-1]])

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        '''
        if isinstance(headers,str):
            headers = [headers]
        
        copy = np.transpose(self.get_all_data())
        indices = self.get_header_indices(headers)
        
        if rows == []:
            new = np.zeros(shape=(len(indices),self.get_num_samples()))
            for i in range(len(indices)):
                new[i] = copy[indices[i]]
        else:
            new = np.zeros(shape=(len(indices),len(rows)))
            for i in range(len(indices)):
                for j in range(len(rows)):
                    new[i][j] = copy[indices[i]][rows[j]]
                    
        return np.transpose(new)
        
