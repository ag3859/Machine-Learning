--[[
Sample Main File
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file shows an example of using the tile utilities.
]]

require('xlua')
require("image")
dofile("tile.lua")
dofile("kmeans.lua")
dofile("mog.lua")

xlua.log('output buildings.txt')
-- An example of using tile
function main()
   -- Read file
   im = tile.imread('buildings.png')
   -- Convert to 7500*64 tiles representing 8x8 patches
   t = tile.imtile(im,{8,8})
	t1=t:size()[1]

   tchange1=t:clone()
   tchange3=t:clone()
--[[   print("tsize")
   print(t:size())
	io.read()

   print("tchange size")
   print(tchange:size())
	io.read()
]]
   --print(t)
   --io.read()
--	print(t[3])
--	io.read()
--	print(t)
--	print("1")
--	io.read()
	
	k={2,4,64,256,8}

	for i=1,5 do

--	print(t:size())
--	print("t:size()")
--	io.read()

		km=kmeans(t:size()[2],k[i])


		Rij=km:learn(tchange1)


--[[		for j=1,k[i] do
			print(km[i].m)
		end
]]

--[[		print(t1)
		print("that is size of t1")
		print(k[i])
		print("value of k[i]")
		print(Rij:size())
		print("size of Rij")
]]
		error=0
   		for j=1,t1 do
			for kloop=1,k[i] do
				if (Rij[j][kloop]==1) then
					error = error + (torch.norm(tchange1[j]-km[kloop].m))^2
					tchange3[j]=km[kloop].m
--					print("tile change number")
--					print(j)
--					tchange[j]=km[kloop]
					break
				end
			end
--			tchange[j]=km[kloop]
		end
		error=error/t1
	   -- Convert back to 800*600 image with 8x8 patches
	   im2 = tile.tileim(tchange3,{8,8},{600,800})
	   -- Show the image
	   image.display(im2)
	   -- The following call can save the image
	   -- tile.imwrite(im2,'boat2.png')
		print("Reporting for K Means, for number of clusters = ")
		print(k[i])
		print("mean square error for these clusters is")
		print(error)
		print("\n")
--		io.read()
	end

--	km Rij

	print("\n Calculating Number of Bits\n")
	km:bits(tchange1)

	print("K MEANS OVER. TIME FOR GAUSSIANS.")
--	io.read()

	G=8
	tchange2=t:clone()
	tchange4=t:clone()
	mogm=mog(t:size()[2],G)


        GRij=mogm:learn(tchange2,km,Rij)

                error=0
                for j=1,t1 do
--                        for gloop=1,G do
                                a,b=torch.max(Rij[j],1)
--[[				print("this is the max value")
				print(b)
				io.read()
]]
--				if (gloop==b) then
                                        error = error + (torch.norm(tchange2[j]-mogm[b[1]].m))^2
                                        tchange4[j]=mogm[b[1]].m
--                                       break
--                                end
--                        end
--                      tchange[j]=km[kloop]
                end
                error=error/t1


           -- Convert back to 800*600 image with 8x8 patches
           im3 = tile.tileim(tchange4,{8,8},{600,800})
           -- Show the image
           image.display(im3)
           -- The following call can save the image
           -- tile.imwrite(im2,'boat2.png')
                print("Reporting for Gaussians, for number of clusters = ")
                print(G)
                print("mean square error for these clusters is")
                print(error)
--                io.read()
	


end

main()
