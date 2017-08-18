import numpy as np
import theano
import theano.tensor as T
   

class LinearRegressionTheano(object):
    """
    Linear regression model based on Theano library. 
    
    Fitting based on gradient descent with L2 regularization. 
    
    Linear regression create hyperplane (line in 1D space, plane in 2D space),
    based on mean squared error. 

    Parameters
    ----------
    seed : int, None by default
        If not None use number as seed for random generator.

    """
    
    def __init__(self, seed=None):
        
        self.__rng = np.random
        
        if not seed is None:
            self.__rng.seed(seed)
            
    
    def __check(self, X, name):
        """ Check type of data for Theano optimization.
        
        Parameters
        ----------
        X : {array-like}
            Matrix of vector for training.
        name : {string}
            Name of the variable.
        
        Raise
        -------
        RuntimeError
            Raise exception if datatype of variable not good for Theano.
        """
        if X.dtype != theano.config.floatX:
            raise RuntimeError("{0} should have type {1}, not {2}".format(name, theano.config.floatX, X.dtype))
            
            
    def __create_train_function(self, num_samples, num_features):
        """Create and compile fitting function based on
        gradient descient.
        
        Parameters
        ----------
        num_samples : {int}
            Numbers of samples in train set.
        num_features : {int}
            Numbers of features in input data.
            
        Returns
        -------
        T : Theano function
            Function takes as input:
                X : {array}, shape=(num_saples, num_features)
                    Input for regression.
                y : {array} , shape=(num_saples, )
                    Output for regression.
                lamb: {float}
                    Regression coefficient.
                alpha: {float}
                    Coefficient for gradient descient.
            Function updates w and b vectors with respect
            to gradient descient.
            Function return:
                cost : {float}
                    Cost with respect to current w and b.
                cost_with_reg : {float}
                    Cost with respect to current w and b 
                    with L2 regularization.
        """   
        # create random vectors as coefficients for regression
        w = theano.shared(self.__rng.randn(num_features), name="w")
        b = theano.shared(0., name="b")
        
        # create input variables
        inp = T.dmatrix("inp")
        out = T.dvector("out")
        lamb = T.dscalar("lamb")
        alpha = T.dscalar("alpha")
       
        prediction = T.dot(inp, w) + b # Linear fucntion
        
        cost = T.sum(T.pow(prediction - out, 2)) / (2*num_samples) # cost function
        cost_with_reg = cost + lamb*T.mean(T.pow(w, 2)) # cost with regularization
        gw, gb = T.grad(cost_with_reg, [w, b]) # calculate gradient 
        
        # compile Theano function
        train = theano.function(inputs=[inp, out, lamb, alpha],
                                outputs=[cost, cost_with_reg],
                                updates=(
                                         (w, w - alpha*gw),
                                         (b, b - alpha*gb)
                                        )
                               )
        
        self.__w = w
        self.__b = b
        
        return train
            
    def __create_predict_func(self):
        """ Create and compile linear function.
        
        Create
        -------
        predict_theano : Theano function
            Function takes as input:
                X : {array}, shape=(num_saples, num_features)
                    Input for linear regression.
            Function return:
                y : {float}
                    Output of linear regression with respect to X.
        """ 

        inp = T.dmatrix("inp")
        prediction = T.dot(inp, self.__w) + self.__b # Linear fucntion
        
        self.predict_theano = theano.function(inputs=[inp],
                                              outputs=[prediction],
                                             )
     
    def predict(self, X):
        """Predict value with trained model.
        
        Parameters
        ----------
            X : {array}, shape=(num_saples, num_features)
                Input for regression.
            
        Returns
            y : {array}, shape=(num_saples, )
                Output of regression model.
        """   
        return self.predict_theano(X)
 
    
    def fit(self, X, y, reg_coef=0.0, a=0.001, max_iter=1000):
        """Fit model to create linear regression model.
        
        Parameters
        ----------
        X : {array}, shape=(num_saples, num_features)
            Input for fitting.
        y : {array} , shape=(num_saples, )
            Targets for fittings.
        reg_coef: {float}
            Regression coefficient.
        a: {float}
            Coefficient for gradient descient.
            
        Create
        -------
        costs : {array} , shape=(max_iter, )
            Cost on each iteration of gradient descient.
        costs_with_regression : {array} , shape=(max_iter, )
            Cost with regresssion on each iteration of gradient descient.
        """   
        
        self.__check(X, "X")
        self.__check(y, "y")
        
        train = self.__create_train_function(X.shape[0], X.shape[1])
        
        errs = []
        errs_reg = []
        for _ in range(max_iter):
            err, err_reg = train(X, y, reg_coef, a)
            errs.append(err)
            errs_reg.append(err_reg)
        
        self.__create_predict_func()
        self.costs = np.array(errs)
        self.costs_with_regression = np.array(errs_reg)
        
    
    def get_coefficients(self):
        """Get coefficients of linear regression model
                    
        Returns
            w : {array}, shape=(num_features, )
                Coefficients of linear model.
            b : {float}, shape=(num_saples, )
                Bias of linear model.
        """  
        return self.__w.get_value(), float(self.__b.get_value())