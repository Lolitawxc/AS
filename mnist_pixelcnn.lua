require "torch"
require "nn"
require "math"
require "gnuplot"
require "image"
require("MaskedConvolution.lua")

-- create the neural network
local function create_pixelcnn(opt)

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
  model:add(nn.MaskedConvolution('nn', 'A', 1, dim, 7, 7, 1, 1, 3, 3))
  model:add(nn.MaskedConvolution('nn', 'B', dim, dim, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(dim, dim, 1, 1))
  if opt.crit == 'softmax' then
    model:add(nn.SpatialConvolution(dim, 256, 1, 1))
    model:add(nn.Reshape(256, 1, 28, 28))
    -- model:add(nn.View(-1):setNumInputDims(2))
  else
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
    
  print( "Training the network" )
  local criterion = nn.ClassNLLCriterion()

  for iteration=1,maxIterations do
    local index = math.random(dataset.data:size(1)) -- pick example at random
    local input = dataset.data[index]
    local output = dataset.label[index]
    
    criterion:forward(network:forward(input), output)
    
    network:zeroGradParameters()
    network:backward(input, criterion:backward(network.output, output))
    network:updateParameters(learningRate)
  end

end

-- predict with a trained neural network
function predict(predictor, dataset, classes, classes_names)

  local mistakes = 0
  local tested_samples = 0
  
  print( "----------------------" )
  print( "Index Label Prediction" )
  for i=1,dataset.data:size(1) do

    local input  = dataset.data[i]
    local class_id = dataset.label[i]
  
    local responses_per_class  =  predictor:forward(input) 
    local probabilites_per_class = torch.exp(responses_per_class)
    local probability, prediction = torch.max(probabilites_per_class, 1) 

     
    if prediction[1] ~= class_id then
      mistakes = mistakes + 1
      local label = classes_names[ classes[class_id] ]
      local predicted_label = classes_names[ classes[prediction[1] ] ]
      print(i , label , predicted_label )
    end

    tested_samples = tested_samples + 1
  end

  local test_err = mistakes/tested_samples
  print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")

end


----------
-- MAIN --
----------

-- Parsing cmd parameters
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-crit', 'sigmoid', 'use sigmoid or softmax')
local config = cmd:parse(arg)
print(config)

-- MNIST dataset
local mnist = require 'mnist'
local dataset = mnist['test' .. 'dataset']()
dataset.data = dataset.data:reshape(dataset.data:size(1), dataset.data:size(2) * dataset.data:size(3)):double()
print(dataset.data:size(1))
print(dataset.label[1])

-- USPS dataset
-- local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
-- print(testing_dataset:size())

-- Network
local network = create_pixelcnn(config)
print(network)

-- Train
fit(network, dataset, 5, 0.01)
predict(network, dataset, {1,2,3,4,5,6,7,8,9,10}, {'0','1','2','3','4','5','6','7','8','9'})