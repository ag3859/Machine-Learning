--[[
Models implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 09/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you should implement models satisfying the following convention,
so that they can be used by trainers you will implement in trainer.lua.

A model object consists of the following fields:

model.w: the parameter tensor. Will be updated by a trainer.

model:l(x,y): the loss function. Should take regularization into consideration.
Should assume that x and y are both tensors. The return value must be a
scalar number (not 1-dim tensor!)

model:dw(x,y): the gradient function. Should take regularization into
consideration. Should assume that x and y are both tensors. The return value
is a tensor of the same dimension as model.w

model:f(x): the output function. Depending on the model, the output function
is the output of a model prior to passing it into a decision function. For
example, in linear model f(x) = w^T x, or in logistic regression model
f(x) = (exp(w^T x) - 1)/(exp(w^T x) + 1). The output should be a tensor.

model:g(x): the decision function. This will produce a vector that will match
the labels. For example, in binary classification it should return either [1]
or [-1] (usually a thresholding by f(x)). The output should be a tensor. This
output will be used in a trainer to test the error rate of a model.

model:train(datasets, ...): (optional) direct training. If a model can be
directly trained using a closed-form formula, it can be implemented here
so that we do not need any trainer for it. Additional parameter is at your
choice (e.g., regularization).

The way I would recommend you to program the model above is to write a func-
tion which returns a table containing the fields above. As an example, a
linear regression model (modLinReg) is provided.

For additional information regarding regularizer, please refer to
regularizer.lua.

For additional information regarding the trainer, please refer to trainer.lua

]]

-- Linear regression module: f(x) = w^T x
-- inputs: dimension of inputs; r: a regularizer
function modLinReg(inputs, r)
   local model = {}
   -- Generate a weight vector initialized randomly
   model.w = torch.rand(inputs)

   -- Define the loss function. Output is a real number (not 1-dim tensor!).
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      return (torch.dot(model.w,x) - y[1])^2/2 + r:l(model.w)
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      return x*(torch.dot(model.w,x) - y[1]) + r:dw(model.w)
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      return torch.ones(1)*torch.dot(model.w,x)
   end
   -- Define the indicator function, who gives a binary classification
   function model:g(x)
      if model:f(x)[1] >= 0 then return torch.ones(1) end
      return -torch.ones(1)
   end
   -- Train directly without a trainer. Should return average loss and
   -- error on the training data
   function model:train(dataset)
      -- Remove the following line and add your stuff
--      print("You have to define this function by yourself! direct soln in modlinreg"..dataset:features());
	local w1=torch.zeros(dataset:features());
	local temp=dataset:features();
	local w2=torch.zeros(temp,temp);
	model.w = torch.zeros(dataset:features())
	local x1=torch.zeros(temp,temp)
	local x2=torch.zeros(temp,temp)
	local w3=torch.zeros(temp,temp)
	temp=0
      
	for i=1,dataset:size() do
		for j=1,dataset:features() do
			x1[j][1]=dataset[i][1][j]
--			x2[j][1]=dataset[i][1][j]		
		end

		x2=x1:transpose(1,2)
--		x2=x2:transpose(1,2)  			   ----X^T
		w2 = w2 + torch.mm(x1,x2)		   ---summation of X*X^T
---		w2:addmm(x1,x2)
		
		temp2=dataset[i][2]				---output y

		w1 = w1 + dataset[i][1]*temp2[1]		---summation of y*X
	end

	w3=torch.inverse(w2)					---inverse of (X*X^T)
	
	model.w=torch.mv(w3,w1)					
        
        --FIND ERROR
	-- Average loss
      local loss = 0
      -- Counter for wrong classifications
      local error = 0
      -- Iterate over all the datasets
      for i = 1,dataset:size() do
         -- Iterative loss averaging        
         loss = loss*(i-1)/i + model:l(dataset[i][1], dataset[i][2])/i
         -- Iterative error rate computation
         if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
            error = error*(i-1)/i
         else
            error = (error*i-error + 1)/i
         end
      end
      -- Return the loss and error ratio
      return loss, error
   end

   -- Train directly without a trainer. Should return average loss and
   -- error on the training data. This takes regularizer into account
   function model:train_reg(dataset)
      -- Remove the following line and add your stuff
--      print("You have to define this function by yourself! direct soln in modlinreg"..dataset:features());
	local w1=torch.zeros(dataset:features());
	local temp=dataset:features();
	local w2=torch.zeros(temp,temp);
	model.w = torch.zeros(dataset:features())
	local x1=torch.zeros(temp,temp)
	local x2=torch.zeros(temp,temp)
	local w3=torch.zeros(temp,temp)
	temp=0
      
	for i=1,dataset:size() do
		for j=1,dataset:features() do
			x1[j][1]=dataset[i][1][j]
--			x2[j][1]=dataset[i][1][j]		
		end

		x2=x1:transpose(1,2)
--		x2=x2:transpose(1,2)  			   ----X^T
		w2 = w2 + torch.mm(x1,x2)		   ---summation of X*X^T
---		w2:addmm(x1,x2)
		
		temp2=dataset[i][2]				---output y

		w1 = w1 + dataset[i][1]*temp2[1]		---summation of y*X
	end

	w3=torch.inverse(w2+(torch.ones(w2:size())*r:lvalue()))					---inverse of (X*X^T)
	
	model.w=torch.mv(w3,w1)					
        
        --FIND ERROR
	-- Average loss
      local loss = 0
      -- Counter for wrong classifications
      local error = 0
      -- Iterate over all the datasets
      for i = 1,dataset:size() do
         -- Iterative loss averaging        
         loss = loss*(i-1)/i + model:l(dataset[i][1], dataset[i][2])/i
         -- Iterative error rate computation
         if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
            error = error*(i-1)/i
         else
            error = (error*i-error + 1)/i
         end
      end
      -- Return the loss and error ratio
      return loss, error
   end



   -- Return this model
   return model
end

-- Perceotron module: f(x) = w^T x
-- inputs: dimension of inputs; r: a regularizer
function modPercep(inputs, r)
   local model = {}
   -- Generate weight vector initialized randomly
   model.w = torch.rand(inputs)
   -- Define the loss function. Output is areal number (not 1-dim tensor!)
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      -- Remove the following line and add your stuff
      return (model:g(x)[1]-y[1])*(torch.dot(model.w,x)) + r:l(model.w)
--      print("You have to define this function by yourself! ok, Written");
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      -- Remove the following line and add your stuff
--      print("we reached here")
      return x*(model:g(x)[1] - y[1]) + r:dw(model.w)
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      -- Remove the following line and add your stuff
      return torch.ones(1)*torch.dot(model.w,x)
   end
   -- Define the indicator function, who gives a binary classification
   function model:g(x)
      -- Remove the following line and add your stuff
      if model:f(x)[1] >= 0 then return torch.ones(1) end
      return -torch.ones(1)
   end
   -- Return this model
   return model
end

-- Logistic regression module: f(x) = (exp(w^T x) - 1)/(exp(w^T x) + 1)
-- inputs: dimension of inputs; r: a regularizer
function modLogReg(inputs, r)
   -- Remove the following line and add your stuff
    local model = {}
   -- Generate weight vector initialized randomly
   model.w = torch.rand(inputs)
   -- Define the loss function. Output is areal number (not 1-dim tensor!)
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      -- Remove the following line and add your stuff
      return (2*torch.log(1+torch.exp(-1*y[1]*torch.dot(model.w,x)))) + r:l(model.w)
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      -- Remove the following line and add your stuff
         return x*(model:f(x)[1] - y[1]) + r:dw(model.w)
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      -- Remove the following line and add your stuff
      return (torch.ones(1)*(torch.exp(torch.dot(model.w,x))-1)/(torch.exp(torch.dot(model.w,x))+1))
   end
   -- Define the indicator function, who gives a binary classification
   function model:g(x)
      -- Remove the following line and add your stuff     NOT USED IN MODELOGREG
      if model:f(x)[1] >= 0 then return torch.ones(1) end
      return -torch.ones(1)
   end
   -- Return this model
   return model

end

-- Multinomial logistic regression module f(x)_k = exp(w_k^T x) / (\sum_j exp(w_j^Tx))
-- inputs: dimension of inputs; classes: number of classes; r: a regularizer
function modMulLogReg(inputs, classes, r)
   local model = {}
   model.w = torch.rand(classes, inputs) 
   
	local temp1=0
	local temp2=0
	
	function model:l(x,y)
		temp1 = 0
      		temp2 = y[1]
      		for i = 1, classes do
      			temp1 = temp1 + torch.exp(torch.dot(model.w[i],x))
      		end   
      		return (torch.log(temp1/(torch.exp(torch.dot(model.w[temp2],x))))) + r:l(model.w)
	end

   function model:dw(x,y)
      temp1 = 0
      temp2 = y[1]
      for i = 1, classes do
      		temp1 = temp1 + torch.exp(torch.dot(model.w[i],x)) 
      end 
      mexp = torch.exp(torch.dot(model.w[temp2],x))
      lwzero = torch.zeros(model.w:size())
      for i = 1, classes do
        	if i == temp2 then 
			lwzero[i] = x*(mexp/temp1-1) 
		else lwzero[i] = x*(mexp/temp1) 
		end
      end

      return (lwzero + r:dw(model.w))
   end

   function model:f(x)
      vones = torch.ones(classes)  
      temp1 = 0     
      for i = 1, classes do
		temp1 = temp1 + torch.exp(torch.dot(model.w[i],x))
      end
      for i = 1, classes do
      		vones[i] = (torch.exp(torch.dot(model.w[i],x)))/temp1
      end

      return vones
   end
   
   function model:g(x)
      max = 0 
      index = 1 
      v = model:f(x)
      for i = 1, classes do
          if v[i] >= max then 
          max = v[i]
          index =i 
          end    
      end
      return torch.ones(1)*index
   end
   
   return model
end
