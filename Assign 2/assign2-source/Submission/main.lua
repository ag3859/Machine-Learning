--[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 10/10/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file contains sample of experiments.
--]]

--require 'xlua'

--xlua.log('test_d_7.txt')

-- Load required libraries and files
dofile("spambase.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")
dofile("kernel.lua")
dofile("crossvalid.lua")
dofile("xsvm.lua")
dofile("mult.lua")
dofile("mnist.lua")

-- An example of using xsvm
function main()

--   local data_train, data_test = spambase:getDatasets(2000,1000)
   local data_train_m, data_test_m = mnist:getDatasets(6000,1000)
	print(data_train:size())
	print(data_test:size())
--[[   for d=1,7 do

	print("\nFor D="..d)
--	print("\n")

--	local k=4 

		local c=2^(-2)
		local marsv=0
		local totsv=0
		local model, error_train, error_test = crossvalid(xsvm.vectorized, 10, data_train, c, d)
	
		print("\t C="..c)
--		print("\n")

--		print(model[2].a)		

		local avgcrossvalidation=0
		local avgtrainerror=0
		local error_actual_test_array={}
		local error_actual_test=0

		for i=1,10 do
			avgcrossvalidation=(avgcrossvalidation*(i-1)+error_test[i])/i
			avgtrainerror=(avgtrainerror*(i-1)+error_train[i])/i

                        error_actual_test_array[i]=model[i]:test(data_test)
                        error_actual_test=(error_actual_test*(i-1)+error_actual_test_array[i])/i

		--	marsv=0
			for j=1,data_train:size() do
				if(model[i].a[j]~=nil) then
--					print("for ith model, jth data, a is")
 --       	                        print(model[i].a[j])
					totsv=totsv+1
					if((model[i].a[j]>0)and(model[i].a[j]<c)) then
						marsv=marsv+1
--						print("for ith model, jth data, a is ")
--	                                        print(model[i].a[j])
--						print(" and hence is a marginal support vector")
					end
				end
			end
--			print("# of marginal support vectors for this model is "..marsv)
                end
		print("\t Avg number of marginal support vectors "..(marsv/10))

		print("\t Avg number of support vectors "..(totsv/10))

		print("\t Avg Cross Valid Error "..avgcrossvalidation)
--		print("\n")
		print("\t Avg Training Error    "..avgtrainerror)
--		print("\n")
		print("\t Avg Testing Error     "..error_actual_test)
	
--		print("\n")
--		print("model.a                   "..model.a)

--		print("\n")

    end

	print("lets start multiclass classification\n")

		local primemodel=multOneVsAll(modPrimSVM)
		local trainerror=primemodel:train(data_train_m)
		local testerror=primemodel:test(data_test_m)
		print("error training "..trainerror)
		print("test error "..testerror)

		local primemodel2=multOneVsOne(modPrimSVM)
		local trainerror2=primemodel2:train(data_train_m)
		local testerror2=primemodel2:test(data_test_m)
		print("error training "..trainerror2)
		print("test error "..testerror2)
]]
end

main()
