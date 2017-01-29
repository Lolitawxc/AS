require 'torch'
require 'nn'
require 'math'
require 'gnuplot'
require 'image'
require 'rnn'
require 'MaskedConvolution'
--require 'RowLSTM'

---------------------------------
------------ NETWORK ------------
---------------------------------

-- create the neural network
local function create_pixelcnn(opt)
  -- opt: model parameters in dictionary

  local function check(model)
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
  model:add(nn.MaskedConvolution('A', 1, dim, 7, 7, 1, 1, 3, 3)) -- 7x7 conv, mask A
  
  if opt.model == 'cnn' then
    model:add(nn.MaskedConvolution('B', dim, dim, 3, 3, 1, 1, 1, 1)) -- 3x3 conv, mask B (can be more than one layer)
  elseif opt.model == 'row' then
    model:add(nn.RowLSTM(dim))
    -- We have to add masked convolutions below into our RowLSTM module.
    -- model:add(nn.MaskedConvolution('B', dim, dim, 3, 1, 1, 1, 1, 1)) -- Input-State : 3x1 conv, mask B
    -- model:add(nn.MaskedConvolution(nil, dim, dim, 3, 1, 1, 1, 1, 1)) -- State-State : 3x1 conv, no mask
  elseif opt.model == 'diag' then
    -- model:add(nn.DiagonalBiLSTM()) -- TODO
    -- We have to add masked convolutions below into our DiagonalBiLSTM module.
    -- model:add(nn.MaskedConvolution('B', dim, dim, 1, 1, 1, 1, 1, 1)) -- Input-State : 1x1 conv, mask B
    -- model:add(nn.MaskedConvolution(nil, dim, dim, 1, 2, 1, 1, 1, 1)) -- State-State : 1x2 conv, no mask
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

  -- check(model)

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
    
  print('Training the network')
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
  print('Network trained')

end

-------------------------------------
------ GENERATION / COMPLETION ------
-------------------------------------

function generate(model)
  -- Generates an image from an empty image.

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

  image.display{image=sample:reshape(imageSize,imageSize), zoom=20}
   image.save('bonres.jpg', sample:reshape(imageSize,imageSize))
end

function complete(model, sample)
  -- Complete an image from its first half.

  local imageSize = 28
  sample = torch.reshape(sample,imageSize,imageSize):type(model._type)

  -- print('init input', sample)

  -- Hide the bottom half of the image
  for i=imageSize/2,imageSize do
    sample[{ i, {} }] = 0
  end

  -- image.display{image=sample:reshape(imageSize,imageSize), zoom=20}

  -- Try to find the hidden part
  for i=imageSize/2,imageSize do
    for j=1,imageSize do
      new_sample = model:forward(sample):reshape(imageSize,imageSize)
      -- print('output', new_sample)
      sample[{ i, j }] = new_sample[{ i, j }]
      -- print('result', sample)
    end
  end

  image.display{image=sample:reshape(imageSize,imageSize), zoom=20}
   image.save('bonres_comp.jpg', sample:reshape(imageSize,imageSize))
end

--------------------------
---------- MAIN ----------
--------------------------

-- Parsing cmd parameters
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-crit', 'sigmoid', 'use sigmoid or softmax')
cmd:option('-model', 'cnn', 'row: RowLSTM, diag:DiagonalBiLSTM, cnn: PixelCNN')
cmd:option('-complete', false, 'complete image otherwise generate new image')
local config = cmd:parse(arg)
-- print(config)

-- USPS dataset
-- local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
-- print(testing_dataset:size())

-- MNIST dataset
local mnist = require 'mnist'
local dataset = mnist['test' .. 'dataset']() -- test set for the moment (less memory cost)
dataset.data = dataset.data:double():div(256)
dataset.data = dataset.data:reshape(dataset.data:size(1), dataset.data:size(2) * dataset.data:size(3)):double()
-- print(dataset.data:size(1))
-- print(dataset.label[1])

-- Network
local network = create_pixelcnn(config)
print(network)

-- Train
fit(network, dataset, 100, 0.01)
if config.complete == false then
    generate(network)
else
    complete(network,dataset.data[5])
end
