require "torch"
require "nn"
require "math"
require "gnuplot"
require "image"

-- global variables
learningRate = 0.01
maxIterations = 100000

function create_pixelrnn()
  model = "pixel_rnn" -- name of model [pixel_rnn, pixel_cnn]
  hidden_dims = 16 -- dimension of hidden states of LSTM or Conv layers
  recurrent_length = 7 -- the length of LSTM or Conv layers
  out_hidden_dims = 32 -- dimesion of hidden states of output Conv layers
  out_recurrent_length = 2 -- the length of output Conv layers
  data = "usps" -- name of dataset [usps, mnist, cifar]
  height, width, channel = 16,16,1

  print("Building starts -> " .. model)
  local prnn = nn.Sequential()


  -- input of main reccurent layers : 7x7 conv, mask A
  print("Building input layer")
  prnn:add( nn.MaskedConvolution(channel, hidden_dims * 2, 7, 7, "A") )

  
  -- -- main reccurent layers : DiagonalBiLSTM layers
  -- for idx=1,recurrent_length do
  --   print("Building DiagonalBiLSTM layer " .. idx)
  --   prnn:add( diagonal_bilstm() ) -- TODO
  -- end


  -- output reccurent layers : ReLU followed by 1x1 conv, mask B (2 layers)
  for idx=1,out_recurrent_length do
    print("Building output layer" .. idx)
    prnn:add( nn.ReLU() )
    prnn:add( nn.MaskedConvolution(channel, out_hidden_dims, 1, 1, "B") )
  end


  -- Sigmoid
  print("Building Sigmoid")
  prnn:add( nn.Sigmoid() )

  print("Building finished -> " .. model)

  return prnn

end


---------------------------
--- CUSTOM CONVOLUTION ----
---------------------------
local MaskedConvolution, parent = torch.class('nn.MaskedConvolution', 'nn.SpatialConvolution')

-- see : https://github.com/torch/nn/blob/master/doc/convolution.md

-- nInputPlane: The number of expected input planes in the image given into forward().
-- nOutputPlane: The number of output planes the convolution layer will produce.
-- kW: The kernel width of the convolution
-- kH: The kernel height of the convolution
-- mask : type of mask [A,B,None]
function MaskedConvolution:__init(nInputPlane, nOutputPlane, kW, kH, mask, dW, dH, padW, padH)
  parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  self.mask = mask
  self:applyMask()
end

-- Apply the mask, if there is one, to the weights.
-- No mask means traditional spatial convolution then nothing more is done to the weights.
function MaskedConvolution:applyMask()
  if self.mask == "A" then
    -- TODO
  elseif self.mask == "B" then
    -- TODO
  end
end


----------------------
--- BASELINE CODE ----
----------------------
--------- || ---------
--------- || ---------
--------- \/ ---------

-- here we set up the architecture of the neural network
function create_network(nb_outputs)

  local ann = nn.Sequential();  -- make a multi-layer structure

  -- 16x16x1   
  ann:add(nn.SpatialConvolution(1,6,5,5))   -- becomes 12x12x6
  ann:add(nn.SpatialSubSampling(6,2,2,2,2)) -- becomes  6x6x6 
    
  ann:add(nn.Reshape(6*6*6))
  ann:add(nn.Tanh())
  ann:add(nn.Linear(6*6*6,nb_outputs))
  ann:add(nn.LogSoftMax())

  return ann

end

-- train a Neural Netowrk
function train_network(network, dataset)
    
  print( "Training the network" )
  local criterion = nn.ClassNLLCriterion()

  for iteration=1,maxIterations do
    local index = math.random(dataset:size()) -- pick example at random
    local input = dataset[index][1]  
    local output = dataset[index][2]
    
    criterion:forward(network:forward(input), output)
    
    network:zeroGradParameters()
    network:backward(input, criterion:backward(network.output, output))
    network:updateParameters(learningRate)
  end

end

function test_predictor(predictor, test_dataset, classes, classes_names)

  local mistakes = 0
  local tested_samples = 0
  
  print( "----------------------" )
  print( "Index Label Prediction" )
  for i=1,test_dataset:size() do

    local input  = test_dataset[i][1]
    local class_id = test_dataset[i][2]
  
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

-- main routine
function main()

  local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
  -- local network = create_network(#classes)
  local prnn = create_pixelrnn()
  print(prnn)


  -- print(testing_dataset[1][1])
  -- -- gnuplot.imagesc(testing_dataset[1][1]:reshape(16,16))
  -- image.display{image=testing_dataset[1][1]:reshape(16,16), zoom=20}


  -- train_network(network, training_dataset)
  -- test_predictor(network, testing_dataset, classes, classes_names)

end

main()
















