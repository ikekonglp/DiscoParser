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
params = cmd:parse(arg)
function make_network(config)
   local input1 = nn.Identity()()
   local input2 = nn.Identity()()

   local network1 = tdnn.build_yoon(params)
   local network2 = tdnn.build_yoon(params)
   
   local u1 = network1(input1)
   local u2 = network2(input2)
   
   local pen = nn.JoinTable(2)({u1, u2})
   local H = config.hiddenSize
   local penultimate_drop = nn.Dropout(0.5)(pen)
   -- return nn.gModule({input1, input2}, {penultimate_drop})
   local penultimate2 = nn.ReLU()(nn.Linear(6*H, H)(penultimate_drop))


   local output = nn.LogSoftMax()(nn.Linear(H, params.labels)(penultimate2))
   
   return nn.gModule({input1, input2}, {output})
end

local f = hdf5.open(params.trainData,'r')
local train_data = {}
local test_data = {}
train_data[1] = f:read('train_arg1'):all():cuda()
train_data[2] = f:read('train_arg2'):all():cuda()
local target_data = f:read('train_label'):all():cuda()
params.labels = target_data:max()


test_data[1] = f:read('dev_arg1'):all():cuda()
test_data[2] = f:read('dev_arg2'):all():cuda()
local test_target_data = f:read('dev_label'):all():cuda()


local criterion = nn.ClassNLLCriterion()

-- shuffle = torch.randperm(train_data:size(1))
-- train_data = train_data:index(1, shuffle:long())
-- target_data = target_data:index(1, shuffle:long())

-- train_data = train_data:cuda()
-- target_data = target_data:cuda()
-- test_data = test_data:cuda()
-- test_target_data = test_target_data:cuda()

network = make_network(params)
network:cuda()
criterion:cuda()

for i = 1, params.epochs do
   -- shuffle = torch.randperm(train_data:size(1))
   -- train_data = train_data:index(1, shuffle:long())
   -- target_data = target_data:index(1, shuffle:long())

   local function g(data, i, size)
      return {data[1]:narrow(1, i, size), 
              data[2]:narrow(1, i, size)}
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
