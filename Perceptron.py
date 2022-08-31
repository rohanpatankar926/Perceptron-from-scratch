class Perceptron:
    def __init__(self,eta:float=None,epochs:int=None):
        self.weights=np.random.randn(3)*1e-4 #small random weights
        trainings= eta is not None and epochs is not None
        if trainings:
            print(f"initial weight before training is-->{self.weights}")
        self.eta=eta
        self.epochs=epochs

    def _z_outcome(self,inputs,weights):
        return np.dot(inputs,weights)
    
    def activation_function(self,z):
        return np.where(z >0 ,1,0)
    
    def fit(self,X,y):
        self.x=X
        self.y=y
        x_with_bias=np.c_[self.x,-np.ones((len(self.x),1))]
        print(f"x with bias: {x_with_bias}")
        
        for epoch in range(self.epochs):
            print(f"for epoch >> {epoch+1}")
            z=self._z_outcome(x_with_bias,self.weights)
            y_hat=self.activation_function(z)
            print(f"Predicted value after forward pass: {y_hat}")
            self.error=self.y-y_hat
            print(f"error: {self.error}")
            self.weights=self.weights+self.eta*np.dot(x_with_bias.T,self.error)
            print(f"Updated weights after epoch: {epoch}/{self.epochs}: \n{self.weights} ")
    
    def predict(self,test_inputs):
        x_with_bias=np.c_[self.x,-np.ones((len(test_inputs),1))]
        z=self._z_outcome(x_with_bias,self.weights)
        return self.activation_function
    
    def loss(self):
        total_loss=np.sum(self.error)
        print(f"Total loss: {total_loss}\n")
        return total_loss
    
    def _create_dir_return_path(self,model_dir,filename):
        os.makedirs(model_dir,exist_ok=True)
        return os.path.join(model_dir,filename)
    
    def save(self,filename,model_dir):
        if model_dir is not None:
            model_file_path=self._create_dir_return_path(model_dir,filename)
            joblib.dump(self,model_file_path)
        else:
            model_file_path=self._create_dir_return_path("model",filename)
            joblib.dump(self,model_file_path)
            
    def load(self,file_path):
        return joblib.load(file_path)
    
def prepare_data(df,target):
    x=df.drop(target,axis=1)
    y=df[target]
    return x,y


AND_GATE={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}
df=pd.DataFrame(AND_GATE)
X,y=prepare_data(df,"y")

if  __name__=="__main__":
	eta=0.001 
	epochs=10
	model_and=Perceptron(eta=eta,epochs=epochs)
	model_and.fit(X,y)
	_=model_and.loss()
