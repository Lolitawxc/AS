require "torch"
require "nn"
require "math"
require "gnuplot"
require "image"

----------------------------------
------- MASKED CONVOLUTION -------
----------------------------------

-- see : https://github.com/torch/nn/blob/master/doc/convolution.md
local MaskedConvolution, Parent = torch.class('nn.MaskedConvolution', 'nn.Module')

function MaskedConvolution:__init(data_type, mask_type,
                                 nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,
                                 verbose)
   -- data_type: 'cudnn' or 'nn'
   -- mask_type: 'A' or 'B' or nil (none)
   -- this module is masked before weight updates,
   -- technically, we apply a mask to 'gradOutput' before 'accGradParameters()'

   Parent.__init(self)

   if mask_type then
      self.m = torch.Tensor(kH, kW):fill(1)
      if ((kH + 1) / 2 < kH) then
         self.m[{{(kH + 1)/2, kH}, {(kW + 1)/2 + 1, kW}}] = 0
         self.m[{{(kH + 1)/2 + 1, kH}, {1, kW}}] = 0
      end
      if mask_type == 'A' then
         self.m[(kH + 1)/2][(kW + 1)/2] = 0
      end
   end

   if self.verbose then
      print('mask', self.m)
   end

   self.mask = torch.Tensor(nOutputPlane, nInputPlane, kH, kW) -- not use forloop in lua to avoid slowdown
   for i = 1, nOutputPlane do
      for j = 1, nInputPlane do
         self.mask[i][j] = self.m
      end
   end

   if data_type == 'cudnn' then
   else
      self.conv = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   end

   local w, gw = self.conv:parameters()
   self.weight = w[1]
   self.bias = w[2]
   self.kH = kH
   self.kW = kW
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.verbose = verbose

end

function MaskedConvolution:updateOutput(input)
   self:clearWeight()
   self.output = self.conv:forward(input)
   return self.output
end

function MaskedConvolution:updateGradInput(input, gradOutput)
   self.gradInput = self.conv:updateGradInput(input, gradOutput)
   return self.gradInput
end

function MaskedConvolution:accGradParameters(input, gradOutput, scale)
   self.conv:accGradParameters(input, gradOutput, scale)
end

function MaskedConvolution:clearWeight()
   if self.verbose then
      print("clear weight of Mask Conv!")
   end
   self.weight:cmul(self.mask)
end

function MaskedConvolution:reset()
end

function MaskedConvolution:zeroGradParameters()
   self.conv:zeroGradParameters()
end

function MaskedConvolution:updateParameters(learningRate)
   self.conv:updateParameters(learningRate)
end

---------------------------------
------------ NETWORK ------------
---------------------------------

-- create the neural network
local function create_pixelcnn(opt)
  -- opt: model parameters in dictionary

  local function testModel(model)
    local imageSize = 28
    local input = torch.randn(1,1,imageSize,imageSize):type(model._type)
    print('input', input)
    print('forward output',{model:forward(input)})
    print('output', model.output)
    print('backward output',{model:backward(input,model.output)})
    model:reset()
  end

  local dim = 16
  local model = nn.Sequential()
  model:add(nn.Reshape(1, 28, 28))
  model:add(nn.MaskedConvolution('nn', 'A', 1, dim, 7, 7, 1, 1, 3, 3)) -- 7x7 conv, mask A
  
  if opt.model == 'cnn' then
    model:add(nn.MaskedConvolution('nn', 'B', dim, dim, 3, 3, 1, 1, 1, 1)) -- 3x3 conv, mask B (can be more than one layer)
  elseif opt.model == 'row' then
    -- model:add(nn.RowLSTM()) -- TODO
    model:add(nn.MaskedConvolution('nn', 'B', dim, dim, 3, 1, 1, 1, 1, 1)) -- 3x1 conv, mask B
    model:add(nn.MaskedConvolution('nn', nil, dim, dim, 3, 1, 1, 1, 1, 1)) -- 3x1 conv, no mask
  elseif opt.model == 'diag' then
    -- model:add(nn.DiagonalBiLSTM()) -- TODO
    model:add(nn.MaskedConvolution('nn', 'B', dim, dim, 1, 1, 1, 1, 1, 1)) -- 1x1 conv, mask B
    model:add(nn.MaskedConvolution('nn', nil, dim, dim, 1, 2, 1, 1, 1, 1)) -- 1x2 conv, no mask
  end

  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(dim, dim, 1, 1))
  
  if opt.crit == 'softmax' then
    model:add(nn.SpatialConvolution(dim, 256, 1, 1))
    model:add(nn.Reshape(256, 1, 28, 28))
    -- model:add(nn.View(-1):setNumInputDims(2))
  elseif opt.crit == 'sigmoid' then
    model:add(nn.View(-1):setNumInputDims(3))
    model:add(nn.Linear(28*28*dim, 28*28))
    model:add(nn.Sigmoid())
  end

  -- testModel(model)

  -- init
  for k,v in pairs(model:findModules'nn.Linear') do
    v.bias:zero()
  end

  for k,v in pairs(model:findModules('nn.MaskedConvolution')) do
    local n = v.kW*v.kH*v.nInputPlane
    v.weight:normal(0,math.sqrt(2/n))
    if v.bias then 
       v.bias:zero() 
    end
  end

  return model
end

-- train a neural network
function fit(network, dataset, maxIterations, learningRate)
    
  print("Training the network")
  local criterion = nn.BCECriterion() -- others to try : nn.MultiLabelSoftMarginCriterion(), nn.ClassNLLCriterion() ; or on CUDA : cudnn.VolumetricCrossEntropyCriterion()

  for iteration=1,maxIterations do
    local index = math.random(dataset.data:size(1)) -- pick random example
    local x = dataset.data[index]
    local y = dataset.data[index] -- unsupervised learning

    output = network:forward(x)
    -- print(x:size())
    -- print(y:size())
    -- print(output:size())
    cost = criterion:forward(output, y)
    -- print(cost)
    delta = criterion:backward(output, y)
    
    network:zeroGradParameters()
    network:backward(x, delta)
    network:updateParameters(learningRate)
  end
  print("Network trained")

end

--------------------------------
---------- GENERATION ----------
--------------------------------

function generate(model)
  local imageSize = 28
  local sample = torch.Tensor(imageSize,imageSize):zero():type(model._type)
  --print('init input', sample)

  for i=1,imageSize do
    for j=1,imageSize do
      new_sample = model:forward(sample):reshape(imageSize,imageSize)
      -- print('output', new_sample)
      sample[{ i, j }] = new_sample[{ i, j }]
      -- print('result', sample)
    end
  end

  image.display{image=new_sample:reshape(imageSize,imageSize), zoom=20}
end

--------------------------
---------- MAIN ----------
--------------------------

-- Parsing cmd parameters
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-crit', 'sigmoid', 'use sigmoid or softmax')
cmd:option('-model', 'cnn', 'row: RowLSTM, diag:DiagonalBiLSTM, cnn: PixelCNN')
local config = cmd:parse(arg)
print(config)

-- USPS dataset
-- local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
-- print(testing_dataset:size())

-- MNIST dataset
local mnist = require 'mnist'
local dataset = mnist['test' .. 'dataset']() -- test set for the moment (less memory cost)
dataset.data = dataset.data:reshape(dataset.data:size(1), dataset.data:size(2) * dataset.data:size(3)):double()
print(dataset.data:size(1))
print(dataset.label[1])

-- Network
local network = create_pixelcnn(config)
print(network)

-- Train
fit(network, dataset, 1000, 0.01)
generate(network)