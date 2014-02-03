--[[
Model file
By Aditya Garg (aditya.garg [at] nyu.edu) @ New York University
Version 0.1, 10/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

--]]

-- Load required libraries and files
dofile("isolet.lua")
dofile("RBF.lua")
dofile("Mulpos.lua")
dofile("NegExp.lua")
require "nn"

function modtwolayer(x,y, train, test)
	
	x=x-1+1
	y=y-1+1

	local mlp = nn.Sequential()
	mlp:add( nn.Linear(x, y) ) 

	mlp:add( nn.Tanh() ) 
	mlp:add( nn.Linear(y, 26) )
	mlp:add( nn.LogSoftMax())

	local criterion=nn.ClassNLLCriterion()

	local trainer = nn.StochasticGradient(mlp, criterion)

	trainer.maxIteration = 50

	trainer.learningRate = 0.1

	trainer:train(train)
	local error = 0
	for i = 1,test:size() do
          val,ind = torch.max(mlp:forward(test[i][1]),1)
          if torch.sum(torch.ne(ind, test[i][2])) == 0 then
             error = error*(i-1)/i
          else
             error = (error*i-error + 1)/i
          end
	end

	print("Error Calculated "..error)
end


function modLogistic(x, train, test)
	x = x + 1 - 1
	
	local mlp = nn.Sequential()
	mlp:add( nn.Linear(x,26))
	mlp:add( nn.LogSoftMax())

	local criterion = nn.ClassNLLCriterion()

	local trainer = nn.StochasticGradient(mlp, criterion)

	trainer.maxIteration = 50

	trainer.learningRate = 0.1

	trainer:train(train)

	local error = 0
	for i = 1,test:size() do
          val,ind = torch.max(mlp:forward(test[i][1]),1)
          if torch.sum(torch.ne(ind, test[i][2])) == 0 then
             error = error*(i-1)/i
          else
             error = (error*i-error + 1)/i
          end
	end

	print("Error Calculated "..error)

end

function modRBF(x, y, train, test)

	local mlp = nn.Sequential()
	mlp:add(nn.RBF(x,y))
	mlp:add(nn.Mulpos(y))
	mlp:add(nn.NegExp())
	mlp:add(nn.Linear(y,26))
	mlp:add(nn.LogSoftMax())
	
	local criterion = nn.ClassNLLCriterion()

	local trainer = nn.StochasticGradient(mlp, criterion)

	trainer.maxIteration = 50

	trainer.learningRate = 0.1

	trainer:train(train)

	local error = 0
	for i = 1,test:size() do
          val,ind = torch.max(mlp:forward(test[i][1]),1)
          if torch.sum(torch.ne(ind, test[i][2])) == 0 then
             error = error*(i-1)/i
          else
             error = (error*i-error + 1)/i
          end
	end

	print("Error Calculated "..error)

end
