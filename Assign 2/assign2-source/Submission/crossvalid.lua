--[[
Cross-validation implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 10/08/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you should implement a cross-validation mechanism that will be
used to perform multiple training on a model. You can implement it in anyway
you want, but the following is how I implemented it:

crossvalid(mfunc, k, dataset)
in which mfunc is a callable that creates a model with train and test methods,
k is the number of folds, dataset is the dataset to be used. dataset must
be randomized (our spambase and mnist datasets automatically do so when calling
getDatasets()).

The returned parameters are models, errors_train and erros_test.

models is a table in which models[i] should store the ith model returned by
calling mfunc.

errors_train is a torch tensor of size k indicating training errors returned by
model[i]:train(dataset).

errors_test is a torch tensor of size k indicating testing errors returned
by model[i]:test(dataset) after training it.

--]]

-- How I implemented cross validation:
-- k: number of folds;
-- mfunc: a callable that creates a model with train() and test() methods.
-- model.train(dataset) should train a model and return the training error
-- model.test(dataset) should return the testing error
-- The return list is: models, errors_train, errors_test where
-- models is a table in which models[k] indicates the kth one returned by mfunc
-- errors_train is a vector of size k indicating the training errors
-- errors_test is a vector of size k indicating the cross-validation errors

dofile("spambase.lua")

function crossvalid(mfunc, k, dataset, c, d)
   -- Remove the following line and add your stuff
--   print("You have to define this function by yourself!");

--	local datatrain, datatest = spambase:getDatasets(100,100)
	local datasetsize=dataset:size()
	local error_train={}
	local error_test={}	
	local train={}
	local test={}

	local trainsize=(datasetsize*(k-1))/k
	local testsize=(datasetsize/k)

	
--	print("dataset size is ")
--	print(datasetsize)
--	print("K fold is ")
--	print(k)


	function train:size() return trainsize end
	function test:size() return testsize end

	local model = {}

		for i=1,k do 
	--		print(i)
	---		1st training set
			local x=1
			local y=1
	--		print("train1")
			for j=1,((i-1)*(datasetsize/k)) do
	--			print(j)
				train[x]={dataset[j][1]:clone(), dataset[j][2]:clone()}
				x=x+1
			end	
	--		print("test")
			for j=((i-1)*(datasetsize/k))+1,(i*(datasetsize/k)) do
	--			print(j)
				test[y]={dataset[j][1]:clone(), dataset[j][2]:clone()}
				y=y+1
			end
	--		print("train2")
			for j=(i*(datasetsize/k)),datasetsize do
	--			print(j)
				train[x]={dataset[j][1]:clone(), dataset[j][2]:clone()}
				x=x+1
			end

	--		print("whats wrong here")
	--		print(train:size())
	--		print(test)
	--		print("Serious prob")
			model[i]=mfunc{kernel=kernPoly(1,d), C=c}
			error_train[i] = (model[i]:train(train))
			error_test[i] = (model[i]:test(test))
		end	
	return model, error_train, error_test
end
