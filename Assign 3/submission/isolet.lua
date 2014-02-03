--[[
Isolet dataset implementation
By Aditya Garg (Aditya.Garg [at] nyu.edu) @ New York University
Version 0.1, 10/04/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In general, all you need to do is to load this file with Isolet.data
presented in you current directory as follows:
t7> dofille "Isolet.lua"
then, you can split out shuffled and normalized training and testing data by
calling Isolet:getDatasets(train_size,test_size), for example:
t7> train, test = Isolet:getDatasets(3000,1000)

The sets train and test (and even Isolet itself) follow the datasets
convention defined in torch tutorial http://www.torch.ch/manual/tutorial/index
, and I quote it here:
"A dataset is an object which implements the operator dataset[index] and
implements the method dataset:size(). The size() methods returns the number of
examples and dataset[i] has to return the i-th example. An example has to be
an object which implements the operator example[field], where field often
takes the value 1 (for input features) or 2 (for corresponding labels), i.e
an example is a pair of input and output objects."

For example, using train[3][1], you get the inputs of the third training
example which is a 58-dim vector, where the last dimension is constantly 1 so
you do not need to worry about the bias in a linear model. Using train[3][2],
you get the output of the  third training example which is a 1-dim vector
whose sole element can only be +1 or -1.
]]

-- the Isolet dataset
isolet = {};

-- The dataset has 4601 rows (observations) 
function isolet:size() return 7797 end

-- Each row (observaton) has 57 features
function isolet:features() return 617 end

-- We have 26 classes, where the digit i is class (i+1).
function isolet:classes() return 26 end

-- Read csv files from the Isolet.data
function isolet:readFile(train_size, test_size)
   -- CSV reading using simple regular expression :)
   local file = 'isolet1+2+3+4.data'
   local file2 = 'isolet5.data'
  

--   print("#1")
   local fp = assert(io.open (file))
   local fp1=assert(io.open (file2))

   local csvtable = {}
   local csvtable1 = {}

--   print("#2")
   for line in fp:lines() do
      local row = {}
      for value in line:gmatch("[^,]+") do
	 -- note: doesn\'t work with strings that contain , values
	 row[#row+1] = value
      end
      csvtable[#csvtable+1] = row
   end
 
--   print("#3")  
   for line1 in fp1:lines() do
      local row = {}
      for value in line1:gmatch("[^,]+") do
         -- note: doesn\'t work with strings that contain , values
         row[#row+1] = value
      end
      csvtable1[#csvtable1+1] = row
   end


   -- Generating random order
   local rorder = torch.randperm(train_size)
   local rorder1 = torch.randperm(test_size)


--   print("#4")

   -- iterate over rows of train data
   for i = 1, train_size do	
      -- iterate over columns (1 .. num_features)
      local input = torch.Tensor(isolet:features())
      local output = torch.Tensor(1)
--	print("#5")
      for j = 1, isolet:features() do
	 -- set entry in feature matrix
	 input[j] = csvtable[i][j]
      end
--	print("#6")
      -- get class label from last column (num_features+1)
      output[1] = csvtable[i][isolet:features()+1]


--[[      -- it should be class -1 if output is 0
	print("#7")
      if output[1] == 0 then output[1] = -1 end
]]
      -- Shuffled dataset
--	print("#8")
      isolet[rorder[i]] = {input, output}
   end

--  print("reached here")
   --iterate over rows of test data
   for i = 1, test_size do
      -- iterate over columns (1 .. num_features)
      local input = torch.Tensor(isolet:features())
      local output = torch.Tensor(1)
      for j = 1, isolet:features() do
         -- set entry in feature matrix
         input[j] = csvtable1[i][j]
      end
      -- get class label from last column (num_features+1)
      output[1] = csvtable1[i][isolet:features()+1]
--[[      -- it should be class -1 if output is 0
      if output[1] == 0 then output[1] = -1 end
]]
    
      -- Shuffled dataset
      isolet[train_size+rorder1[i]] = {input, output}
   end

end



-- Split the dataset into two sets train and test
-- isoset:readFile() must have been executed
function isolet:split(train_size, test_size)
   local train = {}
   local test = {}

	

   function train:size() return train_size end
   function test:size() return test_size end
   function train:features() return isolet:features() end
   function test:features() return isolet:features() end
   function train:classes() return isolet:classes() end
   function test:classes() return isolet:classes() end

   -- iterate over rows
   for i = 1,train:size() do
      -- Cloning data instead of referencing, so that the datset can be split multiple times
      train[i] = {isolet[i][1]:clone(), isolet[i][2][1]}
--	print(train[i])
   end
   -- iterate over rows
--   print(isolet:size())
--	print(train_size)
--	print(test_size)
   for i = 1,test:size() do
      -- Cloning data instead of referencing
--	print(i)
      test[i] = {isolet[i+train:size()][1]:clone(), isolet[i+train:size()][2][1]}
   end

   return train, test
end

-- Normalize the dataset using training set's mean and std
-- train and test must be returned from isolet:split
function isolet:normalize(train, test)
--	print("noramalize"..train[1000])
   -- Allocate mean and variance vectors
   local mean = torch.zeros(train:features())
   local var = torch.zeros(train:features())
   -- Iterative mean computation
   for i = 1,train:size() do
      mean = mean*(i-1)/i + train[i][1]/i
   end
   -- Iterative variance computation
   for i = 1,train:size() do
      var = var*(i-1)/i + torch.pow(train[i][1] - mean,2)/i
   end
   -- Get the standard deviation
   local std = torch.sqrt(var)

   -- If any std is 0, make it 1
   std:apply(function (x) if x == 0 then return 1 end end)

   -- Normalize the training dataset
   for i = 1,train:size() do
      train[i][1] = torch.cdiv(train[i][1]-mean, std)
   end

   -- Normalize the testing dataset
   for i = 1,test:size() do
      test[i][1] = torch.cdiv(test[i][1]-mean, std)
   end

   return train, test
end



-- Add a dimension to the inputs which are constantly 1
-- This is useful to make simple linear modules without thinking about the bias
function isolet:appendOne(train, test)
   -- Sanity check. If dimensions do not match, do nothing.
   if train:features() ~= isolet:features() or test:features() ~= isolet:features() then
      return train, test
   end

--   print("append"..train[1000])

   -- Redefine the features() functions
   function train:features() return isolet:features() + 1 end
   function test:features() return isolet:features() + 1 end
   -- Add dimensions
   for i = 1,train:size() do
      train[i][1] = torch.cat(train[i][1], torch.ones(1))
   end
   for i = 1, test:size() do
      test[i][1] = torch.cat(test[i][1], torch.ones(1))
   end
   -- Return them back
   return train, test
end

-- Get the train and test datasets
function isolet:getIsoletDatasets(train_size, test_size)
   -- If file not read, read the files
   if isolet[1] == nil 
   then print("Reading IsoLet")
	isolet:readFile(train_size, test_size) 
   end
   -- Split the dataset
   print("splitting train and test data")
   local train, test = isolet:split(train_size, test_size)
   -- Normalize the dataset
   print("Normalize data")
   train, test = isolet:normalize(train, test)
   -- Append one to each input
   print("Append Bias")
   train, test = isolet:appendOne(train, test)
   -- return train and test datasets
   return train, test
end



-- Normalize the whitened dataset using training set's mean and std
function isolet:wnormalize(train, test, k)
   -- Allocate mean and variance vectors
   local mean = torch.zeros(k)
   local var = torch.zeros(k)
   -- Iterative mean computation
--   print("normalize begins")
--[[
	print(train[1][1])
	print("train[1][1]")
	io.read()
	]]
   for i = 1,train:size() do
      mean = mean*(i-1)/i + train[i][1]/i
   end
--   print("size check ok")
--	print("1")
   -- Iterative variance computation
   for i = 1,train:size() do
      var = var*(i-1)/i + torch.pow(train[i][1] - mean,2)/i
   end

--	print("2")

   -- Get the standard deviation
   local std = torch.sqrt(var)

   -- If any std is 0, make it 1
   std:apply(function (x) if x == 0 then return 1 end end)
--	print("3")
   -- Normalize the training dataset
   for i = 1,train:size() do
      train[i][1] = torch.cdiv(train[i][1]-mean, std)
   end
--	print("4")
   -- Normalize the testing dataset
   for i = 1,test:size() do
      test[i][1] = torch.cdiv(test[i][1]-mean, std)
   end

   return train, test
end

