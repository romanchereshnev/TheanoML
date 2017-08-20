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
        
    
    def get_coefficients(self):
        """Get coefficients of linear regression model
                    
        Returns
            w : {array}, shape=(num_features, )
                Coefficients of linear model.
            b : {float}, shape=(num_saples, )
                Bias of linear model.
        """  
        return self.__w, float(self.__b)
    
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
        
        num_features = X.shape[1]
        
        def fn(cost, cost_with_reg, w, b, train_x, train_y, a, l):
            """Itration function for Theano scan function. Function implement 
            iteration of gradient descent
            Parameters
            ----------
            costs : {float} 
                Cost on previous gradient descient iteration.
            costs_with_regression : {float} 
                Cost with regresssion on previous gradient descient.
            w : {array}, shape=(num_features)
                Vector of parameters for linear regression
            b : {float} 
                Bias of linear regression
            train_x : {array}, shape=(num_saples, num_features)
                Input for fitting.
            train_y : {array} , shape=(num_saples, )
                Targets for fittings.
            a: {float}
                Coefficient for gradient descient.
            l: {float}
                Regression coefficient.
            Return
            -------
            costs : {float}
                Cost on current gradient descient iteration.
            costs_with_regression : {float} 
                Cost with regresssion on current gradient descient.
            w : {array}, shape=(num_features)
                Updated vector of parameters for linear regression
            b : {float} 
                Updated bias of linear regression
            """ 
            prediction = T.dot(train_x, w) + b # Linear fucntion

            cost = T.mean( (prediction - train_y)**2 ) * 2 # cost function
            cost_with_reg = cost + l*T.mean(w**2) # cost with regularization

            gw, gb = T.grad(cost_with_reg, [w, b]) # calculate gradient 

            w = w - a*gw # update w with gradint
            b = b - a*gb # update b with gradint

            return cost, cost_with_reg, w, b
        
        # create imput values for Theano
        data_x = T.dmatrix()
        data_y, w = T.dvectors("data_y", "w")
        b, cost, cost_with_reg, alpha, lambd = \
            T.dscalars("b", "cost", "cost_with_reg", "alpha", "lambd")
        
        # Loop function for gradient descent
        result, updates = theano.scan(fn=fn, n_steps=max_iter, 
                                      outputs_info=[cost, cost_with_reg, w, b],   
                                      non_sequences=[data_x, data_y, alpha, lambd] 
                                     )

        # Output of loop 
        costs, cost_with_regs, ws, bs = result
        f = theano.function(inputs=[cost, cost_with_reg,  w, b, data_x, data_y, alpha, lambd], 
                            outputs=[costs, cost_with_regs, ws[-1], bs[-1]], 
                            updates=updates)
        

        self.costs, self.costs_with_regression, self.__w, self.__b = \
            f(0.0, 0.0, self.__rng.randn(num_features), 0.0, X, y, a, reg_coef)
        
        self.__create_predict_func()
