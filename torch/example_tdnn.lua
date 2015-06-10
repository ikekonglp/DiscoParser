require 'sys'
require 'hdf5'
train = require 'train'
tdnn = require 'tdnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('TD-NN training')
cmd:text()

train.config(cmd)
tdnn.config(cmd)

cmd:option('-trainData', 'n', 'training data file')
cmd:option('-epochs', '1', 'epochs')
cmd:option('-s', 3, 'renorm value')
-- cmd:option('-testData', '', 'test data file')
params = cmd:parse(arg)
params.addSoftMax = true

local f = hdf5.open(params.trainData,'r')
local train_data = f:read('train'):all()
local target_data = f:read('target'):all()

local test_data = f:read('test'):all()
local test_target_data = f:read('test_target'):all()
local w2v = f:read('w2v'):all()

local network = tdnn.build_yoon(params, w2v)
local criterion = nn.ClassNLLCriterion()
shuffle = torch.randperm(train_data:size(1))
train_data = train_data:index(1, shuffle:long())
target_data = target_data:index(1, shuffle:long())

train_data = train_data:cuda()
target_data = target_data:cuda()
test_data = test_data:cuda()
test_target_data = test_target_data:cuda()

network:cuda()
criterion:cuda()

for i = 1, params.epochs do
   shuffle = torch.randperm(train_data:size(1))
   train_data = train_data:index(1, shuffle:long())
   target_data = target_data:index(1, shuffle:long())
   -- network:apply(function (layer)
   --                  if layer.weight ~= nil and layer.bias == nil then 
   --                     torch.renorm(layer.weight, 2, 1, params.s)
   --                  end
   --               end)

   local function g(data, i, size)
      return data:narrow(1, i, size)
   end

   train.train(network, 
               criterion, 
               train_data,
               target_data, g, params)
   train.eval(network, criterion, 
              test_data,
              test_target_data, g, 
              params)
end
-- 
