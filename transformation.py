'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Iris Liu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import analysis
import data
import math


class Transformation(analysis.Analysis):

    def __init__(self, data_orig, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        data_orig: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables
            — `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`
        '''
        analysis.Analysis.__init__(self, data)
        self.data_orig = data_orig
        self.data = data

    def project(self, headers):
        '''Project the data on the list of data variables specified by `headers` — i.e. select a
        subset of the variables from the original dataset. In other words, populate the instance
        variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list dictates the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables are optional.
        '''
        dict = {}
        for i in range(len(headers)):
            dict[headers[i]] = i
        self.data = data.Data(headers = headers, data = self.data_orig.select_data(headers), header2col = dict)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.
        '''
        matrix = self.data.get_all_data()
        N = matrix.shape[0]
        column = np.ones([N, 1])
        return np.hstack((matrix, column))

    def translation_matrix(self, headers, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        M = len(headers)
        translation = np.eye(M+1)
        for i in range(M):
            translation[i,M] = magnitudes[i]
        return translation

    def scale_matrix(self, headers, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The scaling matrix.
        '''
        M = len(headers)
        scale = np.eye(M+1)
        for i in range(M):
            scale[i,i] = magnitudes[i]
        return scale

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data about the ONE
        axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.
        '''
        headers = self.data.get_headers()
        radians = np.radians(degrees)

        R_x = np.array([[                 1,                 0,                 0,                 0],
                        [                 0, math.cos(radians),-math.sin(radians),                 0],
                        [                 0, math.sin(radians), math.cos(radians),                 0],
                        [                 0,                 0,                 0,                 1]
                        ])
                        
        R_y = np.array([[ math.cos(radians),                 0, math.sin(radians),                 0],
                        [                 0,                 1,                 0,                 0],
                        [-math.sin(radians),                 0, math.cos(radians),                 0],
                        [                 0,                 0,                 0,                 1]
                        ])

        R_z = np.array([[ math.cos(radians),-math.sin(radians),                 0,                 0],
                        [ math.sin(radians), math.cos(radians),                 0,                 0],
                        [                 0,                 0,                 1,                 0],
                        [                 0,                 0,                 0,                 1]
                        ])

        if header == headers[0]:
            return R_x
        elif header == headers[1]:
            return R_y
        else:
            return R_z

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected dataset after it has been transformed by `C`
        '''
        temp = C @ self.get_data_homogeneous().transpose()
        return temp.transpose()

    def translate(self, headers, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''
        
        newDict = self.data.get_mappings()
        
        translation = self.translation_matrix(headers, magnitudes)
        newData = translation @ self.get_data_homogeneous().transpose()
        
        self.data = data.Data(headers = headers, data = newData.transpose()[:,:-1], header2col = dict)
        return newData.transpose()[:,:-1]

    def scale(self, headers, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''
        newDict = self.data.get_mappings()
        
        scale = self.scale_matrix(headers, magnitudes)
        newData = scale @ self.get_data_homogeneous().transpose()
        
        self.data = data.Data(headers = headers, data = newData.transpose()[:,:-1], header2col = dict)
        return newData.transpose()[:,:-1]

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''
        
        headers = self.data.get_headers()
        newDict = self.data.get_mappings()
        
        newData = self.get_data_homogeneous()
        radians = np.radians(degrees)
        
        rotation = self.rotation_matrix_3d(header, degrees)
        newData = rotation @ self.get_data_homogeneous().transpose()
        
        self.data = data.Data(headers = headers, data = newData.transpose()[:,:-1], header2col = dict)
        return newData.transpose()[:,:-1]

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        
        headers = self.data.get_headers()
        
        newData = self.data.get_all_data()
        min, max = newData.min(), newData.max()
        
        trans_magnitudes = []
        for i in range(len(headers)):
            trans_magnitudes.append(-1*min)
            
        self.translate(headers, trans_magnitudes)
        
        scale_magnitudes = []
        for i in range(len(headers)):
            scale_magnitudes.append(1/(max-min))
        self.scale(headers, scale_magnitudes)
        
        return self.data.get_all_data()
        

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        
        min, _range = self.min(headers), self.range(headers)
        self.translate(headers, -1*min)
        
        scale_magnitudes = []
        for i in range(len(headers)):
            scale_magnitudes.append(1/_range[i])
        self.scale(headers, scale_magnitudes)
        
        return self.data.get_all_data()
        
    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        x_data = self.data.select_data([ind_var])
        y_data = self.data.select_data([dep_var])
        c_data = self.data.select_data([c_var])
        fig = plt.scatter(x_data, y_data, c = c_data, s = 60, marker='o', edgecolors='darkgray')
        
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.colorbar().set_label(c_var)
        plt.title(title)
        
        return fig
        

    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap)

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls )
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax
        
    def scatter_size_color(self, ind_var, dep_var, s_var, c_var, title=None):
        '''Creates a 2D scatter plot with a size representing 3rd dimension
        a color scale representing the 4th dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        s_var: Header of the variable that will be plotted according to size.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        x_data = self.data.select_data([ind_var])
        y_data = self.data.select_data([dep_var])
        s_data = self.data.select_data([s_var])
        c_data = self.data.select_data([c_var])
        fig = plt.scatter(x_data, y_data, s = 100*s_data, c = c_data,
                          cmap='plasma', marker='o', edgecolors='darkgray')
        
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.colorbar().set_label(c_var)
        plt.title(title, fontsize=20, pad=20)
        
        return fig

    def normalize_together_zscore(self):
        '''Normalize all variables in the projected dataset together by translating the mean
        (across all variables) to zero and scaling its standard deviation to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        newDict = self.data.get_mappings()
        
        newData = self.data.get_all_data()
        mean, std_dev = newData.mean(), newData.std()
        newData = (newData - mean)/std_dev
        
        self.data = data.Data(headers = headers, data = newData, header2col = dict)
        
        return newData

    def normalize_separately_zscore(self):
        '''Normalize each variable separately by translating the mean to zero and scaling
        its standard devation to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        newDict = self.data.get_mappings()
        
        newData = self.data.get_all_data()
        mean, std_dev = newData.mean(axis=0), newData.std(axis=0)
        newData = (newData - mean)/std_dev
        
        self.data = data.Data(headers = headers, data = newData, header2col = dict)
        
        return newData

    def normalize_together_vectorization(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        newDict = self.data.get_mappings()
        
        newData = self.data.get_all_data()
        max, min = newData.max(), newData.min()
        newData = (newData - min)/(max - min)
        
        self.data = data.Data(headers = headers, data = newData, header2col = dict)
        
        return newData
        

    def normalize_separately_vectorization(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        newDict = self.data.get_mappings()
        
        newData = self.data.get_all_data()
        max, min = newData.max(axis=0), newData.min(axis=0)
        newData = (newData - min)/(max - min)
        
        self.data = data.Data(headers = headers, data = newData, header2col = dict)
        
        return newData
