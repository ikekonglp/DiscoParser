require 'nn'
require 'cunn'
require 'nngraph'
require 'sys'

tdnn = {}

function tdnn.config(cmd)
   cmd:option('-vocabSize', 0, 'size of vocabulary')
   cmd:option('-embedSize', 300, 'size of embedding vec')
   cmd:option('-nlayers', 3, 'size of tdnn')
   cmd:option('-hiddenSize', 100, 'size of hidden layers')
   cmd:option('-kernelSize', 3, 'size of conv kernel')
   cmd:option('-maxPoolSize', 2, 'size of temporal max kern')
end


function tdnn.build(config)
   local V = config.vocabSize
   local D = config.embedSize
   local L = config.nlayers 
   local H = config.hiddenSize 
   local K = config.kernelSize
   local M = config.maxPoolSize 

   local input = nn.Identity()()
   local embed = nn.LookupTable(V, D)(input)
   local inlayer = embed
   for l = 1, L do 
      local temporal = nn.TemporalConvolution(D, H, K)(inlayer)
      local pool = nn.TemporalMaxPooling(M)(temporal)
      local nonlin = nn.ReLU()(pool)
      inlayer = nonlin
   end
   local penultimate = nn.Max(3)(nn.Transpose({2,3})(inlayer))
   local penultimate_drop = nn.Dropout(0.5)(penultimate)
   local output = nn.LogSoftMax()(nn.Linear(H, 2)(penultimate_drop))
   
   return nn.gModule({input}, {output})
end


function tdnn.build_yoon(config, init_embed)
   local V = config.vocabSize
   local D = config.embedSize
   local L = config.nlayers 
   local H = config.hiddenSize 
   local K = config.kernelSize
   local M = config.maxPoolSize 

   local input = nn.Identity()()
   local embed = nn.LookupTable(V, D)
   if init_embed then
      print(embed.weight:size(), init_embed:size())
      embed.weight:copy(init_embed)
   end
   local inlayer = embed(input)

   local temporal1 = nn.TemporalConvolution(D, H, K)(inlayer)
   local nonlin1 = nn.ReLU()(temporal1)
   local pen1 = nn.Max(3)(nn.Transpose({2,3})(nonlin1))

   local temporal2 = nn.TemporalConvolution(D, H, K+1)(inlayer)
   local nonlin2 = nn.ReLU()(temporal2)
   local pen2 = nn.Max(3)(nn.Transpose({2,3})(nonlin2))

   local temporal3 = nn.TemporalConvolution(D, H, K+2)(inlayer)
   local nonlin3 = nn.ReLU()(temporal3)
   local pen3 = nn.Max(3)(nn.Transpose({2,3})(nonlin3))

   local penultimate = nn.JoinTable(2)({pen1, pen2, pen3})
   if config.addSoftMax then 
      local penultimate_drop = nn.Dropout(0.5)(penultimate)
      local output = nn.LogSoftMax()(nn.Linear(3*H, 2)(penultimate_drop))
      return nn.gModule({input}, {output})
   else
      return nn.gModule({input}, {penultimate})
   end

   

end




return tdnn
