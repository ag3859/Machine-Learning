require "nn"

local NegExp = torch.class('nn.NegExp', 'nn.Module')

function NegExp:updateOutput(input)
   newip = torch.Tensor(input:size());
   newip:copy(input);
   newip:mul(-1)
   return input.nn.Exp_updateOutput(self, newip)
end

function NegExp:updateGradInput(input, gradOutput)
   newop = torch.Tensor(input:size());
   newop:copy(input);
   newop:mul(-1);
   temp = torch.Tensor(gradOutput:size());
   temp:copy(gradOutput);
   temp:mul(-1);
   return input.nn.Exp_updateGradInput(self, newop, temp)
end
