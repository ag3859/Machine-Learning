require "nn"

local RBF, parent = torch.class('nn.RBF', 'nn.Module')

function RBF:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   
   self:reset()
end

function RBF:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end

   -- we do this so the initialization is exactly
   -- the same than in previous torch versions
   for i=1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
                                        return torch.uniform(-stdv, stdv)
                                     end)
      self.bias[i] = torch.uniform(-stdv, stdv)
   end
end

function RBF:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      newop = torch.Tensor(self.bias:size(1)):fill(0)

      for i = 1, self.bias:size(1) do
        for j = 1 , input:size(1) do
          newop[i] =torch.pow((input[j] - self.weight[i][j]),2) + newop[i]
         end
      end
      self.output:add(newop)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.output:resize(nframe, nunit)
      self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)
      self.output:addmm(1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function RBF:updateGradInput(input, gradOutput)
   if self.gradInput then

      if input:dim() == 1 then
         self.gradInput:resizeAs(input)
         newip = torch.Tensor(self.bias:size(1),input:size(1)):fill(0)
         wt = self.weight
         for i = 1,self.bias:size(1) do
             for j = 1,input:size(1) do 
               newip[i][j] = input[j] - wt[i][j]
             end
         end
        newipt = newip:t()         
         self.gradInput:addmv(0, 2, newipt, gradOutput)
      else
        error('input must be vector')
         
--	print(newip)
--	print("newip")


      end

      return self.gradInput
   end
end

function RBF:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
        newwt = torch.Tensor(self.bias:size(1),input:size(1)):fill(0)
         for i = 1,self.bias:size(1) do
             for j = 1,input:size(1) do
               newwt[i][j] =  2*(self.weight[i][j] - input[j])*gradOutput[i]
             end
         end
      self.gradWeight:add(scale,newwt)
      self.gradBias:add(scale, gradOutput)      
   elseif input:dim() == 2 then
       error('input must be vector')
   end

end

-- we do not need to accumulate parameters when sharing
RBF.sharedAccUpdateGradParameters = RBF.accUpdateGradParameters
