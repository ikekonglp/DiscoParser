require 'nn'

require 'cunn'
require 'nngraph'
require 'sys'

tdnn = {}

function tdnn.config(cmd)
   cmd:option('-vocabSize', 0, 'size of vocabulary')

   -- Used defaults from Yoon's paper.
   cmd:option('-embedSize', 300, 'size of embedding vec')
   cmd:option('-hiddenSize', 100, 'size of hidden layers')
   cmd:option('-kernelSizeA', 3, 'size of conv kernel')
   cmd:option('-kernelSizeB', 4, 'size of conv kernel')
   cmd:option('-kernelSizeC', 5, 'size of conv kernel')

   cmd:option('-L2s', 3, 'renorm value')
end

function tdnn.build(config, init_embed)
   -- Name parameters.
   local V = config.vocabSize
   local D = config.embedSize
   local L = 3 
   local H = config.hiddenSize 
   local K = {config.kernelSizeA,
              config.kernelSizeB,
              config.kernelSizeC}
   local M = config.maxPoolSize 
   local O = config.labels 

   
   local input = nn.Identity()()

   -- Start by embedding and if given, use the passed in (w2v) weights.
   local embed = nn.LookupTable(V, D)
   if init_embed then
      embed.weight:copy(init_embed)
   end
   local inlayer = embed(input)

   -- Now do L (3) convolutions with different kernels. 
   -- Each consists of a temporal conv and a max-over-time.
   local pen = {}
   for i = 1, L do 
      local temporal = nn.TemporalConvolution(D, H, K[i])(inlayer)
      local nonlin = nn.ReLU()(temporal)
      table.insert(pen, nn.Max(3)(nn.Transpose({2,3})(nonlin)))
   end

   -- Concat the input layers add dropout, throw into softmax-.
   local penultimate = nn.JoinTable(2)(pen)
   return nn.gModule({input}, {penultimate})
end


function tdnn.build_pairwise(config, init_embed)
   -- Name parameters.
   local V = config.vocabSize
   local D = config.embedSize
   local H = config.hiddenSize 
   
   local input = nn.Identity()()
   local embed = nn.LookupTable(2*V, D)
   if init_embed then
      -- embed.weight:narrow(1,1,V):copy(init_embed)
      -- embed.weight:narrow(1,V,V):copy(init_embed)
   end
   local inlayer = nn.Transpose({2, 4})(nn.View(30, 30, 2*D)(embed(input)))

   local temporal = nn.SpatialConvolution(2*D, H, 2, 2)(inlayer)
   local nonlin = temporal
   local pen = nn.ReLU()(nn.Max(3)(nn.Max(4)(nonlin)))

   local penultimate = pen
   return nn.gModule({input}, {penultimate})
end


function tdnn.rescale(linear, config)
   local w = linear.weight   
   local n = linear.weight:view(w:size(1)*w:size(2)):norm()
   if (n > config.L2s) then 
      w:mul(config.L2s):div(n)
   end
end

return tdnn
