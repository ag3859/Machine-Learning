--[[
Multi-class classification using binary classifiers implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 09/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you have to implement multOneVsAll and multOneVsOne. As an example
, part of multOneVsAll is given. These functions accept a parameter mfunc,
which is a function. Upon calling mfunc, a trainable model is returned with
whom you can run model:train(dataset) to train and return training error.
model:g(x) should give a classification, and model:l(x,y) should give the loss
on sample x y.

Of course, you can implement everything in your own way and disregard the code
here. 
--]]

-- mfunc is a callable that will return a trainable object
-- The trainable object must have to protocols:
-- model:train(dataset) will train the object. Return values are discarded
-- model:l(x,y) will return the loss on sample x with label y (-1 or 1).
dofile("regularizer.lua")

function multOneVsAll(mfunc)
   -- Create an one-vs-all trainer
   local mult = {}
   -- Transform the dataset for one versus all
   local function procOneVsAll(dataset)
      -- The data table consists of dataset:classes() datasets
      local data = {}
      -- Iterate through each dataset
      for i = 1, dataset:classes() do
	 -- Create this dataset, with size() method returning the same thing as dataset
	 data[i] = {size = dataset.size}
	 -- Modify the labels
	 for j = 1, dataset:size() do
	    -- Create entry
	    data[i][j] = {}
	    -- Copy the input
	    data[i][j][1] = dataset[j][1]
	    if dataset[j][2][1] == i then
	       -- The label same to this class i is set to 1
	       data[i][j][2] = torch.ones(1)
	    else
	       -- The label different from this class i is set to -1
	       data[i][j][2] = -torch.ones(1)
	    end
	 end
      end
      -- Return this set of datsets
      return data
   end
   -- Train models
   function mult:train(dataset)
      -- Define mult:classes
      mult.classes = dataset.classes
      -- Preprocess the data
      local data = procOneVsAll(dataset)
      -- Iterate through the number of classes
      for i = 1, dataset:classes() do
--	print("calling mult trainer")
	 -- Create a model
	 mult[i] = mfunc(dataset:features(),regL2(0.05))
	 -- Train the model
	 mult[i]:train(data[i])
      end
      -- Return the training error
      return mult:test(dataset)
   end
   -- Test on dataset
   function mult:test(dataset)
      -- Set returning testing error
      local error = 0
      -- Iterate through the number of classes
      for i = 1, dataset:size() do
	 -- Iterative error rate computation
	 if torch.sum(torch.ne(mult:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error/i*(i-1)
	 else
	    error = error/i*(i-1) + 1/i
	 end
      end
      -- Return the testing error
      return error
   end
   -- The decision function
   function mult:g(x)
      -- Remove the following line and add your stuff
     -- print("You have to define this function by yourself!");
     maxf = mult[1]:f(x)[1]
     maxclass = 1
     for i =1,10 do
       if maxf < mult[i]:f(x)[1] then
         maxf = mult[i]:f(x)[1]
         maxclass = i 
       end
     end 
    return torch.ones(1)*maxclass
	
   end
   -- Return this one-vs-all trainer
   return mult
end






















-- mfunc is a callable that will return a trainable object
-- The trainable object must have to protocols:
-- model:train(dataset) will train the object. Return values are discarded
-- model:l(x,y) will return the loss on sample x with label y (-1 or 1).
-- model:g(x) will determine the label of a given x.
function multOneVsOne(mfunc)
   -- Remove the following line and add your stuff
--   print("You have to define this function by yourself!");
	
   local mult = {}
   for i = 1, 10 do
    mult[i] = {}
   end
   -- Transform the dataset for one versus all
   local function procOneVsOne(dataset)
      -- The data table consists of dataset:classes() datasets
      local data = {}
      -- Iterate through each dataset
      for i = 1, dataset:classes() do
	 -- Create this dataset, with size() method returning the same thing as dataset
         data[i] = {}
        for j = i+1, dataset:classes() do
	 data[i][j] = {}
--         data[i][j] = {size = dataset.size}
         s = 1
         size_array= {}
	 -- Modify the labels
	   for k = 1, dataset:size() do
	    -- Create entry
            
	    data[i][j][s] = {}
	    -- Copy the input
            
	    if dataset[k][2][1] == i then
              data[i][j][s][1] = dataset[k][1]
	       -- The label same to this class i is set to 1
	       data[i][j][s][2] = torch.ones(1)
               s = s +1
	    elseif dataset[k][2][1] == j then
               data[i][j][s][1] = dataset[k][1]
	       -- The label different from this class i is set to -1
	       data[i][j][s][2] = -torch.ones(1)
               s = s + 1
	    end
	   end
         data[i][j].count = s-1
         data[i][j].size =  function() return data[i][j].count end
        end
      end 
      -- Return this set of datsets
      return data
   end

   -- Train models
   function mult:train(dataset)
      -- Define mult:classes
      mult.classes = dataset.classes
      -- Preprocess the data
      local data = procOneVsOne(dataset)
      -- Iterate through the number of classes
      for i = 1, dataset:classes() do
       for j= i+1, dataset:classes() do
	 -- Create a model
	 mult[i][j] = mfunc(dataset:features(),regL2(0.05))
	 -- Train the model
	 loss, error = mult[i][j]:train(data[i][j])
--         print("loss and error"..loss..error)
       end
      end
      -- Return the training error
      return mult:test(dataset)
   end
   -- Test on dataset
   function mult:test(dataset)
      -- Set returning testing error
      local error = 0
      -- Iterate through the number of classes
      for i = 1, dataset:size() do
	 -- Iterative error rate computation
	 if torch.sum(torch.ne(mult:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error/i*(i-1)
	 else
	    error = error/i*(i-1) + 1/i
	 end
      end
      -- Return the testing error
      return error
   end
   -- The decision function
   function mult:g(x)
      -- Remove the following line and add your stuff
--      print("909You have to define this function by yourself!")
--     max_f = mult[1]:f(x)[1]
--     max_class = 1
     weight = torch.zeros(10)
     for i =1,10 do
       for j= i+1,10 do
--       print("fx is"..mult[i]:f(x))
         if mult[i][j]:g(x)[1] == 1 then
           weight[i] = weight[i] + 1 
         else
           weight[j]= weight[j] + 1
         end
       end 
     end 
    y,i = torch.max(weight,1)
    max_class = i[1]
    return torch.ones(1)*max_class

   end
   -- Return this one-vs-all trainer
   return mult 



end
