--[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 09/22/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)
]]

-- Load required libraries and files
dofile("spambase.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")
dofile("mnist.lua")

-- This is just an example
function main()
	
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(10,1000)
   local data_train_multinom, data_test_multinom = mnist:getDatasets(6000,1000)

   -- 2. Initialize a linear regression model with l2 regularization (lambda = 0.05) keep it 0.009 for modlinreg wid SGD
   print("\n");

   print("Initializing a Perceptoron model with l2 regularization... ok")
--   local modelPer = modPercep(data_train:features(), regL2(0.05))

   print("Initializing a Linear model with l2 regularization... ok")
--   local modelLinReg_SGD = modLinReg(data_train:features(), regL2(2))

   print("Initializing a Linear model, Direct Soln, with l2 regularization...ok")
--   local modelLinReg_direct = modLinReg(data_train:features(), regL2(0.05))
--   local modelLinReg_direct_reg = modLinReg(data_train:features(), regL2(0.05))

   print("Initializing a Log regression model with l2 regularization...")
--   local modelLogReg_SGD = modLogReg(data_train:features(), regL1(0.01))

   print("Initializing a Mul Log regression model with l2 regularization...")
   local multinomlogreg = modMulLogReg(data_train_multinom:features(),10,regL2(.05))


   -- 3. Initialize a batch trainer with constant step size = 0.05 eta learning rate
   print("\n");

   print("Initializing a perceptron batch trainer with constant step size 0.05... ok")
--   local trainer_per_batch = trainerBatch(modelPer, stepCons(0.05))
   
   print("Initializing a liner SGD trainer with constant step size 0.005... ok")
--   local trainer_lin_SGD = trainerSGD(modelLinReg_SGD, stepCons(0.005))
   
   print("Initializing a logistic SGD trainer with constant step size 0.001... ok")
--   local trainer_log_SGD = trainerSGD(modelLogReg_SGD, stepCons(0.001))

   print("Initializing a SGD trainer with constant step size 0.005... ok") 
   local trainer_multinomlogreg = trainerSGD(multinomlogreg, stepCons(0.005))


   -- 4. Perform training for 100 steps
   print("\n");

   print("Training for 100 batch steps Perceptron... ok")
--   local loss_train_per_batch, error_train_per_batch = trainer_per_batch:train(data_train, 100)

   print("Training for  SGD steps Linear... ok")
--   local loss_train_lin_SGD, error_train_lin_SGD = trainer_lin_SGD:train(data_train, 100)

   print("Implepenting Direct Soln for Liner Reg...no trainer initialized for Direct...with and without RegL2 ok")
--   local loss_train_lin_direct, error_train_lin_direct = modelLinReg_direct:train(data_train)
--   local loss_train_lin_direct_reg, error_train_lin_direct_reg = modelLinReg_direct_reg:train_reg(data_train)

   print("Implepenting SGD for Log Reg... ok")
--   local loss_train_log_SGD, error_train_log_SGD = trainer_log_SGD:train(data_train, 100)

   print("Implepenting SGD for Log Reg... ok")
   local loss_train_multinomlogreg, error_train_multinomlogreg = trainer_multinomlogreg:train(data_train_multinom, 6000)


   -- 5. Perform test using the model
   print("\n");

   print("Testing Perceptron...  ok")
--   local loss_test_per_batch, error_test_per_batch = trainer_per_batch:test(data_test)

   print("Testing Linear Regression SGD...  ok")
--   local loss_test_lin_SGD, error_test_lin_SGD = trainer_lin_SGD:test(data_test)

   print("Testing Log Regression SGD...  ok")
--   local loss_test_log_SGD, error_test_log_SGD = trainer_log_SGD:test(data_test)

   print("Testing Mul Log Regression SGD...  ok")
   local loss_test_multinomlogreg, error_test_multinomlogreg = trainer_multinomlogreg:test(data_test_multinom)

   -- 6. Print the result
   print("\n");

--   print("FOR PERCEPTRON....Training loss = "..loss_train_per_batch..", error = "..error_train_per_batch.."; Testing loss = "..loss_test_per_batch..", error = "..error_test_per_batch)

--   print("\nFOR LINEAR REGRESSION USING SGD...Training loss = "..loss_train_lin_SGD..", error = "..error_train_lin_SGD.."; Testing loss = "..loss_test_lin_SGD..", error = "..error_test_lin_SGD)

--   print("\nFOR LINEAR REGRESSION USING DIRECT SOLN...Training loss = "..loss_train_lin_direct..", error = "..error_train_lin_direct)

--   print("\nFOR LINEAR REGRESSION USING DIRECT SOLN WITH L2 REGULARIZER...Training loss = "..loss_train_lin_direct_reg..", error = "..error_train_lin_direct_reg)

--   print("\nFOR LOGISTIC REGRESSION USING SGD...Training loss = "..loss_train_log_SGD..", error = "..error_train_log_SGD.."; Testing loss = "..loss_test_log_SGD..", error = "..error_test_log_SGD)
  
     print("\nFOR MULTINOMIAL REGRESSION with l2 reg: Training loss = "..loss_train_multinomlogreg..", error = "..error_train_multinomlogreg.."; Testing loss = "..loss_test_multinomlogreg..", error = "..error_test_multinomlogreg)


   print("\n")
end

main()
