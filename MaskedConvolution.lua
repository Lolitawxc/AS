-- see : https://github.com/torch/nn/blob/master/doc/convolution.md
local MaskedConvolution, Parent = torch.class('nn.MaskedConvolution', 'nn.Module')

function MaskedConvolution:__init(mask_type, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  -- mask_type: 'A' or 'B' or nil (no mask)
  -- This module is masked before weight updates. We apply a mask to 'gradOutput' before 'accGradParameters()'.

  Parent.__init(self)

  self.mask = torch.Tensor(nOutputPlane, nInputPlane, kH, kW):fill(1)

  if mask_type then
    center_h = (kH + 1)/2
    center_w = (kW + 1)/2

    if ((kH + 1) / 2 < kH) then
      self.mask[{ {}, {}, { center_h  , kH }, { center_w+1, kW } }] = 0
      self.mask[{ {}, {}, { center_h+1, kH }, { 1         , kW } }] = 0
    end

    if mask_type == 'A' then
      self.mask[{ {}, {}, center_h, center_w }] = 0
    end

  end

  -- print(self.mask[{ 1, 1, {}, {} }])

  -- For the rest it's an usual SpatialConvolution
  self.conv = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

  local w, gw = self.conv:parameters()
  self.weight = w[1]
  self.bias = w[2]
  self.kH = kH
  self.kW = kW
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane

end

function MaskedConvolution:updateOutput(input)
  self:applyMask()
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

function MaskedConvolution:applyMask()
  self.weight:cmul(self.mask)
end

function MaskedConvolution:zeroGradParameters()
  self.conv:zeroGradParameters()
end

function MaskedConvolution:updateParameters(learningRate)
  self.conv:updateParameters(learningRate)
end