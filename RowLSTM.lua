-- see : https://github.com/Element-Research/rnn
local RowLSTM, Parent = torch.class('nn.RowLSTM', 'nn.LSTM')

function RowLSTM:__init(hiddenSize)
   Parent.__init(self, hiddenSize, hiddenSize)
end

function RowLSTM:buildCell()
   -- build
   self.inputGate = self:buildInputGate()
   self.forgetGate = self:buildForgetGate()
   self.hiddenLayer = self:buildHidden()

   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(nn.SelectTable(3))
   forget:add(concat)
   forget:add(nn.CMulTable())

   -- input = inputGate{input, output(t-1), cell(t-1)} * hiddenLayer{input, output(t-1), cell(t-1)}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.hiddenLayer)
   input:add(concat2)
   input:add(nn.CMulTable())

   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
   self.cellLayer = cell
   return cell
end

-- cell(t) = cellLayer{input, output(t-1), cell(t-1)}
-- output(t) = outputGate{input, output(t-1), cell(t)}*tanh(cell(t))
-- output of Model is table : {output(t), cell(t)}
function RowLSTM:buildModel()
   -- build components
   self.cellLayer = self:buildCell()
   self.outputGate = self:buildOutputGate()

   -- assemble
   local concat = nn.ConcatTable()
   concat:add(nn.NarrowTable(1,2)):add(self.cellLayer)
   local model = nn.Sequential()
   model:add(concat)

   -- output of concat is {{input, output}, cell(t)},
   -- so flatten to {input, output, cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   return model
end