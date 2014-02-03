--[[
K-Means clustering algorithm implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The k-means algorithm should be presented here. You can implement it in any way
you want. For your convenience, a clustering object is provided at mcluster.lua

Here is how I implemented it:

kmeans(n,k) is a constructor to return an object km which will perform k-means
algorithm on data of dimension n with k clusters. The km object stores the i-th
cluster at km[i], which is an mcluster object. The km object has the following
methods:

km:g(x): the decision function to decide which cluster the vector x belongs to.
Return a scalar representing cluster index.

km:f(x): the output function to output a prototype that could replace vector x.

km:learn(x): learn the clusters using x, which is a m*n matrix representing m
data samples.
]]

dofile("mcluster.lua")

-- Create a k-means learner
-- n: dimension of data
-- k: number of clusters
function kmeans(n,k)
-- Remove the following line and add your stuff


	local km={}
	km.k=k
	km.m=7500
        local Rij = torch.Tensor(km.m,km.k):fill(0)
	local clusterwise_data = torch.Tensor(km.m,n):fill(0)
	for i = 1,km.k do
		km[i]=mcluster(n)
                --set a random sample as our training example
--                j=torch.rand(64)
--                km[i]:set_m(x[j])
        end



	function km:learn(x)				---for every sample, checks which cluster it can be part of
							---and save the cluster number for every sample number
--[[		local datawise_cluster# = torch.Tensor(km.m):fill(-1)
		for i=1,km.m do
			cluster#=km:g(x[i])
			datawise_cluster#[i,cluster#]=1
		end
]]
--		updateRij()

		j=torch.rand(k)*7500

		j:floor()

	        for i = 1,km.k do
--        	        km[i]=mcluster(n)
                	--set a random sample as our training example
--	                j=torch.rand(64)
		
--			print(x[j[i]])
--			print("2")
--			io.read()
		
--        	        km[i]:set_m(x[j[i]])
			
			km[i]:set_m(x[math.random(7500)])
	        end
		


		local convergence_counter=0
		local Rijnumber=0
		while (convergence_counter ~= km.k) do
			km:updateRij(x)
--			print(Rij)
--			print("Rij updation number ")
			Rijnumber=Rijnumber+1
--			print(Rijnumber)
--			io.read()
			convergence_counter=0
			for i=1,km.k do					---for every cluster
				local index=0
				for j=1,km.m do				---check in every sample
--					print("enters 2nd loop to find all ")
					if(Rij[j][i]==1) then
						index=index+1
--						print("finds a sample in current cluster number"..i)
--[[						print(clusterwise_data[index]:size())
						print("clusterwise_data[index]:size()")

						print(x[j]:size())
						print("x[j]:size()")

]]				--		io.read()

						clusterwise_data[index]=x[j]
					end
				end
		
--				print(clusterwise_data:size())
--				print("clusterwise_data:size()")
--				print("another way to get size")
--				print(index)
--				io.read()
--				r=torch.Tensor(clusterwise_data:size()[1])
				r=torch.Tensor(index):fill(1)
--				print(r:size())
--				print("r:size() in k means")
				
				convergence_counter = convergence_counter + km[i]:learn(clusterwise_data,r,index)
--				updateRij()
--				print("return after learning new means, press enter")
--				io.read()
			end
--			if (convergence_counter==km.k)
--				break;
--			updateRij()
		end
		km:updateRij(x)
		print("Convergence achieved in K Means...generating new image...")
--		io.read()
		return Rij
	end

	function km:updateRij(x)
                for i=1,km.m do
                        local cluster_number=km:g(x[i])
			Rij[i]=Rij[i]*0
                        Rij[i][cluster_number]=1
                end
	end

	function km:f(x)				---returns cluster value for the each vector 	
		local cluster_number = km:g(x)
		return km[cluster_number]
	end

	function km:g(y)				---return cluster number for a sample by checking dist with each cluster
		local min =  99999
		local cluster_number = -1
		for i=1,km.k do
			edist=km[i]:eval(y)
			if edist<min then
				min = edist
				cluster_number = i
			end
		end
		return cluster_number
	end
	
	return km

end
