'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Iris Liu
Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import math


class Analysis:
    def __init__(self, data):
        '''
        Analysis Object Constructor
        
        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        
        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        
        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over,
            or over all indices if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        '''
        
        selected = self.data.select_data(headers, rows)
        mins = selected.min(axis = 0)
        return mins

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        
        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over,
            or over all indices if rows=[]
            
        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables
        '''
        
        selected = self.data.select_data(headers, rows)
        maxs = selected.max(axis = 0)
        return maxs

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        
        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over,
            or over all indices if rows=[]
            
        Returns
        -----------
        ranges: ndarray. shape=(len(headers),)
            range values for each of the selected header variables
        '''
        
        return self.max(headers, rows) - self.min(headers, rows)


    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).
        
        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over,
            or over all indices if rows=[]
            
        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            mean values for each of the selected header variables
        '''
        
        selected = self.data.select_data(headers, rows)
        sum = selected.sum(axis = 0)
        if rows == []:
            return sum/self.data.get_num_samples()
        else:
            return sum/len(selected)

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        
        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over,
            or over all indices if rows=[]
            
        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables
        '''
        
        selected = self.data.select_data(headers, rows)
        diff = (selected - self.mean(headers, rows))**2
        total = diff.sum(axis = 0)
        if rows == []:
            return total/(self.data.get_num_samples()-1)
        else:
            return total/(len(selected)-1)

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        
        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over,
            or over all indices if rows=[]
            
        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables
        '''
        
        return np.sqrt(self.var(headers, rows))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.
        '''
        
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.
        
        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot
        '''
        
        x = self.data.select_data(ind_var)
        y = self.data.select_data(dep_var)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.plot(np.sum(x, axis = 1), np.sum(y, axis = 1), 'o')
        return np.sum(x, axis = 1),np.sum(y, axis = 1)
        

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots
        '''
        
        fig, axes = plt.subplots(len(data_vars), len(data_vars), figsize=fig_sz, sharex=True, sharey=True)
        fig.suptitle(title)
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                x = self.data.select_data(data_vars[i])
                y = self.data.select_data(data_vars[j])
                axes[i,j].plot(y,x,'o')
                if i == len(data_vars)-1:
                    axes[i,j].set_xlabel(data_vars[j])
                if j == 0:
                    axes[i,j].set_ylabel(data_vars[i])
        for ax in axes.flat:
            ax.label_outer()
        
        return fig, axes
