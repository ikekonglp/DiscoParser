require 'optim'

train = {}

function train.config(cmd)
   cmd:option('-learningRate',0.1, 'learning rate')
   cmd:option('-batchSize', 50, 'size of mini-batch')
end

function train.train(network, criterion, data, target, g, config) 
   network:training()
   state = {
      learningRate = config.learningRate,
      weightDecay = 1.0
   }

   -- assert(data:size(1) == target:size(1))
   local loss = 0 
   local total = 0 
   sys.tic()
   x, dl_dx = network:getParameters()
   for i = 1, data[1]:size(1) - config.batchSize, config.batchSize do
      local func = function(x_new)
         network:zeroGradParameters()
         dl_dx:zero()

         if x ~= x_new then 
            x:copy(x_new)
         end
         local input = g(data, i, config.batchSize) 
         local targ = target:narrow(1, i, config.batchSize)
         local out = network:forward(input)
         local local_loss = criterion:forward(out, targ) 
         loss = loss + local_loss * config.batchSize
         total = total + config.batchSize
         local deriv = criterion:backward(out, targ)
         network:backward(input, deriv)
         return local_loss, dl_dx
      end
      optim.adagrad(func, x, state)

      local V = config.vocabSize
      local D = config.embedSize
      w = x:narrow(1, V * D, x:size(1) - V*D)
      
      -- n = w:norm()
      -- if (n > config.s) then 
      --    w:mul(config.s):div(n)
      -- end
   end
   
   print("[EPOCH loss=", loss / total, 
         "time=", sys.toc(), "]")
end


function train.eval(network, criterion, data, target, g, config) 
   network:evaluate()
   -- assert(data:size(1) == target:size(1))
   local loss = 0 
   local total = 0 
   local classes = {}
   for i = 1, config.labels do
      table.insert(classes, i)
   end
   confusion = optim.ConfusionMatrix(classes) 
   confusion:zero()  
   local correct = 0
   sys.tic()
   for i = 1, data[1]:size(1) - config.batchSize, config.batchSize do
      local input = g(data, i, config.batchSize)
      local targ = target:narrow(1, i, config.batchSize)
      -- print(out, targ)
      local out = network:forward(input)
      loss = loss + criterion:forward(out, targ) * config.batchSize
      total = total + config.batchSize
      
      for j = 1, config.batchSize do 
         confusion:add(out[j], targ[j])   
         if (out[j][1] > out[j][2] and targ[j] == 1) or 
         (out[j][1] < out[j][2] and targ[j] == 2) then
               correct = correct + 1
         end
      end
   end
   print(confusion)
   print("[EVAL loss=", loss/total, "time=", sys.toc(), "correct=", correct / total,  "]")
end



return train