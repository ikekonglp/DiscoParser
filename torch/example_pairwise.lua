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

cmd:option('-test', false)
cmd:option('-trainData', 'n', 'training data file')
cmd:option('-epochs', 1, 'epochs')
cmd:option('-labels', 10, 'labels')
cmd:option('-cuda', 1, 'use gpu')
cmd:option('-dropoutP', 0.5, 'dropout rate')
cmd:option('-network', 1)
params = cmd:parse(arg)

function make_network(config, w2v)
   local H = config.hiddenSize

   local input1 = nn.Identity()()
   local input2 = nn.Identity()()
   local input3 = nn.Identity()()

   local network1 = tdnn.build(params, w2v)
   local network2 = tdnn.build(params, w2v)
   local features = nn.LookupTable(40000, H)
   
   local u = {network1(input1),
              network2(input2),
              nn.Mean(2)(features(input3))
   }
   
   local pen = nn.JoinTable(2)(u)

   local penultimate2 = nn.ReLU()(
      nn.Linear(7*H, 3*H)(nn.Dropout(config.dropoutP)(pen)))
   local penultimate_drop = nn.Dropout(config.dropoutP)(penultimate2)
   local output = nn.LogSoftMax()(nn.Linear(3*H, params.labels)(penultimate_drop))
   return nn.gModule({input1, input2, input3}, {output})
end

function make_network2(config, w2v)
   local input3 = nn.Identity()()
   local network3 = tdnn.build_pairwise(params, w2v)
   
   local H = config.hiddenSize

   local penultimate2 = nn.ReLU()(
      nn.Linear(H, H)(nn.Dropout(config.dropoutP)(network3(input3))))
   local penultimate_drop = nn.Dropout(config.dropoutP)(penultimate2)
   local output = nn.LogSoftMax()(nn.Linear(H, params.labels)(penultimate_drop))
   return nn.gModule({input3}, {output})
end

local function make_cuda(vec)
   if params.cuda ~= 0 then 
      return vec:cuda()
   else
      return vec
   end
end

local f = hdf5.open(params.trainData,'r')



if params.network == 1 then
   function g(data, i, size)
      return {data[1]:narrow(1, i, size), 
              data[2]:narrow(1, i, size),
              data[3]:narrow(1, i, size)}
   end
else
   function g(data, i, size)
      a = align:narrow(1, i, size):view(size, 30*30*2)
      return a                  
   end
end
local criterion = nn.ClassNLLCriterion()

if params.test then 
   local target_data = make_cuda(f:read('train_label'):all())--:narrow(1,1,100))
   params.labels = target_data:max()

   local network = torch.load("best_model.torch")

   criterion:cuda()
   local test_data = {}
   test_data[1] = make_cuda(f:read('dev_arg1'):all())
   test_data[2] = make_cuda(f:read('dev_arg2'):all())
   test_data[3] = make_cuda(f:read('dev_parse'):all())
   local test_target_data = make_cuda(f:read('dev_label'):all())
   local score, result = train.eval(network, criterion, test_data,
                                    test_target_data, g, params)
   local out = hdf5.open("result.hd5", 'w')
   out:write("dev_result", torch.Tensor(result))
else 

   local w2v = f:read('embeding'):all()

   local train_data = {}
   local test_data = {}
   train_data[1] = make_cuda(f:read('train_arg1'):all())--:all():narrow(1,1,100))
   train_data[2] = make_cuda(f:read('train_arg2'):all())--:narrow(1,1,100))
   train_data[3] = make_cuda(f:read('train_parse'):all())--:narrow(1,1,100))
   local target_data = make_cuda(f:read('train_label'):all())--:narrow(1,1,100))
   params.labels = target_data:max()
   target_data = target_data

   local n1 = math.min(train_data[1]:size(2), 30)
   local n2 = math.min(train_data[2]:size(2), 30)

   -- -- Alignment Matrix
   -- local align = torch.ones(train_data[1]:size(1), n1, n2, 2):long():fill(2)

   -- local n = 1
   -- for i = 1, n1 do 
   --    for j = 1, n2 do 
   --       align[{{}, i, j, 1}]:copy(train_data[1][{{}, i}])
   --       n = n + 1
   --       align[{{}, i, j, 2}]:copy(train_data[2][{{}, j}] + params.vocabSize) 
   --       n = n + 1
   --    end
   -- end
   --- print(train_data[2][55])

   test_data[1] = make_cuda(f:read('dev_arg1'):all())
   test_data[2] = make_cuda(f:read('dev_arg2'):all())
   test_data[3] = make_cuda(f:read('dev_parse'):all())--:narrow(1,1,100))
   local test_target_data = make_cuda(f:read('dev_label'):all())

   if params.network == 1 then
      network = make_network(params, w2v)
   else
      network = make_network2(params, w2v)
   end

   if params.cuda~=0 then 
      network:cuda()
      criterion:cuda()
      -- align = align:cuda()
   end

   -- L2 Rescaling. 
   local linear = {}
   local embedding = {}
   local function get_linear(layer) 
      local tn = torch.typename(layer)
      if tn == "nn.gModule" then
         layer:apply(get_linear)
      end
      if tn == "nn.Linear" then
         table.insert(linear, layer)
      end
      if tn == "nn.LookupTable" then
         table.insert(embedding, layer)
      end
   end
   network:apply(get_linear)

   local function rescale(x)
      for i = 1, #linear do
         tdnn.rescale(linear[i], params)
      end
      for i = 1, #embedding do
         embedding[i].weight[1]:zero()
      end
   end

   local best_score = 1e10
   state = {
      learningRate = config.learningRate,
   }
   x, dl_dx = network:getParameters()
   for i = 1, params.epochs do
      shuffle = torch.randperm(train_data[1]:size(1))
      train_data[1] = train_data[1]:index(1, shuffle:long())
      train_data[2] = train_data[2]:index(1, shuffle:long())
      train_data[3] = train_data[3]:index(1, shuffle:long())
      target_data = target_data:index(1, shuffle:long())

      train.train(network, x, dl_dx,
                  criterion, 
                  train_data,
                  target_data, g, rescale, params, state)
      local score = train.eval(network, criterion, 
                               test_data,
                               test_target_data, g, 
                               params)
      if score < best_score then
         best_score = score
         print("BEST SCORE", score)
         torch.save("best_model.torch", network)
      end
   end
end
