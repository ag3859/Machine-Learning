function whitenDatasets(train, test, p)

	local trainx = torch.Tensor(train:size(), train:features())
	local testx = torch.Tensor(test:size(), test:features())

	for i = 1, train:size() do
		trainx[i] = train[i][1]:clone()
	end
	for i=1, test:size() do
		testx[i] = test[i][1]:clone()
	end

--	print(trainx[1])

--	print("1")


	local trainxt = trainx:t()
	local testxt = testx:t()

	

--	print("2")

	local trainsigma = torch.Tensor(train:features(),train:features()):fill(0) 
--	local traintemp = torch.Tensor(train:features(),train:features()):fill(0)

--	print("3")
	local trainvec = torch.Tensor(1,train:features()):fill(0)

--	print("4")
	
	local testsigma = torch.Tensor(test:features(),test:features()):fill(0) 
--	local testtemp = torch.Tensor(test:features(),test:features()):fill(0)
	local testvec = torch.Tensor(1,test:features()):fill(0)

--	print("5")

	print("Calculating Co-Variance X^T X")
	
	for i=1, train:size() do

		trainvec[1]=trainx[i]
--		print("6")
--		print(trainvec[1])
		trainsigma:addmm(trainxt:narrow(2,i,1), trainvec)
	
--		trainsigma = traintemp
	end
--	print("7")
	for i=1, test:size() do

		testvec[1]=testx[i]
--		print("8")
		testsigma:addmm(testxt:narrow(2,i,1), testvec)
	
--		testsigma = testtemp
	end


--	print("9")
	trainsigma = trainsigma/(train:size())
	testsigma = testsigma/(test:size())

	print("Running SVD")

	trainu,trains,trainv = torch.svd(trainsigma)	
--	testu, tests, testv = torch.svd(testsigma)               not to be done as per Xiang

--	print("10")

--	print(trainu)
--	print("trainu")
--	io.read()

	local trainut = (trainu:narrow(2, 1, p)):t()
--	local testut = (testu:narrow(2,1,p)):t()		not to be done

	p=p-1+1

	local newtrain = torch.Tensor(p,train:size()):fill(0)
	local newtest = torch.Tensor(p,test:size()):fill(0)

	print("Multiplying U and X")

	newtrain:addmm(trainut,trainxt)
	newtest:addmm(trainut,testxt)			--testut made trainut as we are to take U from train's SVD only
	

	local wtrain = {}
	local wtest = {}

--	print("11")

--[[	print(newtrain:narrow(2,1,1))
	print("newtrain:narrow(2,i,1)")
	print("has to be converted")
	io.read()
]]
	for i = 1, train:size() do
		wtrain[i]={newtrain:narrow(2,i,1):resize(p), train[i][2]}
--		wtrain[i][1]=newtrain:narrow(2,i,1)
--		wtrain[i][2]=train[i][2]
	end
	
--	print("12")

	for i=1, test:size() do
		wtest[i]={newtest:narrow(2,i,1):resize(p), test[i][2]}
--		wtest[i][1]=newtest:narrow(2,i,1)
--		wtest[i][2]=test[i][2]
	end
	
--	local wtrain = train
--	local wtest = test

--	print("13")

	function wtrain:features() return p end
	function wtest:features() return p end
	function wtrain:classes() return 26 end
	function wtest:classes() return 26 end
	function wtrain:size() return train:size() end
	function wtest:size() return test:size() end

	return wtrain, wtest	
end
