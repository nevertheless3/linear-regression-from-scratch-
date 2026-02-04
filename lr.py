import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    @staticmethod
    def cost_func(y, yhat):
        #MSE
        return np.sum((yhat - y) ** 2) / (2 * len(y))  

    def calculate_gradient(self, x, t, y_hat, y):
        #partial derivative of bias and weight in respect to loss(here=MSE)
        cost_derivative = np.mean(y_hat - y) 
        errors = y_hat - y
        if t == 'b':
            return cost_derivative
        else:
            return np.dot(x.T , errors) / len(errors)

    @staticmethod
    def train_test_split(x, y, train_size=0.8):
        mask = np.random.rand(len(x)) < train_size
        train_x, test_x = x[mask], x[~mask]
        train_y, test_y = y[mask], y[~mask]
        return train_x, train_y, test_x, test_y
        

    def fit(self, x , y , w=None, b=None, learning_rate=1e-3, epochs=10000):

        nx = x.shape[1]

        if w is None:
            w = np.random.normal(-0.1,0.1,(nx,))
        if b is None:
            b = np.random.normal(-0.1 , 0.1)

        train_x, train_y, test_x, test_y = self.train_test_split(x , y)

        for epoch in range(epochs):
            y_hat = np.dot(train_x,w) + b
            
            gradient_w = self.calculate_gradient(train_x , 'w', y_hat, train_y)
            gradient_b = self.calculate_gradient(train_x , 'b', y_hat, train_y)
            
            w -= learning_rate *gradient_w
            b -= learning_rate * gradient_b

            y_hat_test = np.dot(test_x , w)+b
            test_loss  = self.cost_func(test_y , y_hat_test)
            training_loss = self.cost_func(train_y ,y_hat )
            
            if epoch % 10 == 0:
                print(f'epoch = {epoch} training loss = {training_loss:.3f} test loss = {test_loss:.3f}')


        return w, b
    


if __name__ == '__main__':
    #example
    lr = LinearRegression()
    x = np.random.uniform(0, 1, (100, 3)) 
    true_w = np.array([2, 3, 4])
    true_b = 6
    y = np.dot(x, true_w) + true_b 

    w, b = lr.fit(x, y)