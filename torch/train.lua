require 'optim'

train = {}

function train.config(cmd)
   cmd:option('-learningRate',0.1, 'learning rate')
   cmd:option('-batchSize', 50, 'size of mini-batch')
end

function train.train(network, criterion, data, target, g, rescale, config, state) 
   network:training()

   local loss = 0 
   local total = 0 
   sys.tic()
   x, dl_dx = network:getParameters()
   local n = 1
   for i = 1, data[1]:size(1) - config.batchSize, config.batchSize do
      n = n + 1
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
      optim.adadelta(func, x, state)
      rescale(x)
      if n % 10 == 0 then
         print(n)
      end
   end

   print("[EPOCH loss=", loss / total, 
         "time=", sys.toc(), "]")
end


function train.eval(network, criterion, data, target, g, config) 
   network:evaluate()
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
   local result = {}
   local n = data[1]:size(1)
   for i = 1, n, config.batchSize do
      local off = config.batchSize
      if i + off > n then 
         off = n - i
      end
      local input = g(data, i, off)
      local targ = target:narrow(1, i, off)
      local out = network:forward(input)
      loss = loss + criterion:forward(out, targ) * off
      total = total + off
      
      for j = 1, off do 
         confusion:add(out[j], targ[j])
         local a, i = out[j]:max(1)
         table.insert(result, i[1])
      end
   end
   print(confusion)
   print("[EVAL loss=", loss/total, "time=", sys.toc(), "]")
   return loss / total, result

end



return train
