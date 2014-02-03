require('torch')
require('libnn')
require('nn')

dofile('RBF.lua')
dofile('NegExp.lua')
dofile('Mulpos.lua')


local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local nntest = {}
local nntestx = {}


function nntest.RBF()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local input = torch.Tensor(ini):zero()
   local module = nn.RBF(ini,inj)


--   print("state testing")
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

--   print("weight testing")
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

--   print("bias testing")
   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')


--  print("weight testing 2")
   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

--  print("bias testing 2")
   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')


   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
	--print("1")
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
	--print("2")
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   
   local ferr,berr = jac.testIO(module,input)
--   print("4ward error")
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
--print("back error")
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function nntest.NegExp()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.NegExp()

   local err = jac.testJacobian(module,input)
--   print("state testing")
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)

--   print("4ward error")
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
--print("back error")
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mulpos()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Mulpos(ini*inj*ink)

   local err = jac.testJacobian(module,input)
-- print("state testing")
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
--print("weight test")
   mytester:assertlt(err,precision, 'error on weight ')
   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
print("weight test2")
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
	--print("1")
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   local ferr,berr = jac.testIO(module,input)
--   print("4ward error")
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
--print("back error")
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


mytester:add(nntest)

if not nn then
   require 'nn'

   jac = nn.Jacobian
   mytester:run()
else
   
   jac = nn.Jacobian

      math.randomseed(os.time())
      mytester:run()  
end
