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
cmd:option('-epochs', 1, 'epochs')
cmd:option('-labels', 10, 'labels')
cmd:option('-s', 3, 'renorm value')
cmd:option('-cuda', 1, 'use gpu')
params = cmd:parse(arg)

function make_network(config)
   local input1 = nn.Identity()()
   local input2 = nn.Identity()()

   local network1 = tdnn.build(params)
   local network2 = tdnn.build(params)
   
   local u1 = network1(input1)
   local u2 = network2(input2)
   
   local pen = nn.JoinTable(2)({u1, u2})
   local H = config.hiddenSize

   local penultimate2 = nn.ReLU()(nn.Linear(6*H, H)(pen))
   local penultimate_drop = nn.Dropout(0.5)(penultimate2)
   local output = nn.LogSoftMax()(nn.Linear(H, params.labels)(penultimate_drop))
   return nn.gModule({input1, input2}, {output})
end

local function make_cuda(vec)
   if params.cuda then 
      return vec:cuda()
   else
      return vec
   end
end

local f = hdf5.open(params.trainData,'r')
local train_data = {}
local test_data = {}
train_data[1] = make_cuda(f:read('train_arg1'):all())
train_data[2] = make_cuda(f:read('train_arg2'):all())
local target_data = make_cuda(f:read('train_label'):all())
params.labels = target_data:max()
target_data = target_data


test_data[1] = make_cuda(f:read('dev_arg1'):all())
test_data[2] = make_cuda(f:read('dev_arg2'):all())
local test_target_data = make_cuda(f:read('dev_label'):all())


local criterion = nn.ClassNLLCriterion()
network = make_network(params)

if params.cuda then 
   network:cuda()
   criterion:cuda()
end

-- L2 Rescaling. 
local linear
local embedding = {}
local function get_linear(layer) 
   local tn = torch.typename(layer)
   if tn == "nn.gModule" then
      layer:apply(get_linear)
   end
   if tn == "nn.Linear" then
      linear = layer
   end
   if tn == "nn.LookupTable" then
      table.insert(embedding, layer)
   end
end
network:apply(get_linear)

local function rescale(x)
   tdnn.rescale(x, linear, params)
   for i = 1, #embedding do
      embedding[i].weight[1]:zero()
   end
end


for i = 1, params.epochs do
   shuffle = torch.randperm(train_data[1]:size(1))
   train_data[1] = train_data[1]:index(1, shuffle:long())
   train_data[2] = train_data[2]:index(1, shuffle:long())
   target_data = target_data:index(1, shuffle:long())


   local function g(data, i, size)
      return {data[1]:narrow(1, i, size), 
              data[2]:narrow(1, i, size)}
   end
   
   train.train(network, 
               criterion, 
               train_data,
               target_data, g, rescale, params)
   train.eval(network, criterion, 
              test_data,
              test_target_data, g, 
              params)
end
