--[[
Mixture of Gaussians Implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The mixture of gaussians algorithm should be presented here. You can implement
it in anyway you want. For your convenience, a multivariate gaussian object is
provided at gaussian.lua.

Here is how I implemented it:

mog(n,k) is a constructor to return an object m which will perform MoG
algorithm on data of dimension n with k gaussians. The m object stores the i-th
gaussian at m[i], which is a gaussian object. The m object has the following
methods:

m:g(x): The decision function which returns a vector of k elements indicating
each gaussian's likelihood

m:f(x): The output function to output a prototype that could replace vector x.

m:learn(x,p,eps): learn the gaussians using x, which is an m*n matrix
representing m data samples. p is regularization to keep each gaussian's
covariance matrices non-singular. eps is a stop criterion.
]]

dofile("gaussian.lua")
dofile("kmeans.lua")

-- Create a MoG learner
-- n: dimension of data
-- k: number of gaussians
function mog(n,k)
-- Remove the following line and add your stuff


	local mogm={}
	mogm.m=7500
	mogm.k=k

	for i=1,mogm.k do
		mogm[i]=gaussian(n)		
	end

	local Rij=torch.Tensor(mogm.m,mogm.k):fill(0)
	local likVal=torch.Tensor(mogm.m,1):fill(0)

	local clusterwise_data = torch.Tensor(mogm.m,n):fill(0)

	local randwt=torch.rand(mogm.k)
	local wtsum=torch.sum(randwt)
	local r=randwt/wtsum
--	print("weight vector ")
--	print(r)
--	io.read()
	function mogm:learn(x,km,Rk)				---for every sample, checks which cluster it can be part of
							---and save the cluster number for every sample number
--		j=torch.rand(k)*7500

--		j:floor()
		Rij=Rk
	        for i = 1,mogm.k do
			mogm[i]:set_m(km[i].m)
	        end
		
--		print("set random kmean to Gmeans points")
		print("Max -ve likelihood function value before Gaussian starts, using K Means values")
           	mogm:likelihood(x)
		local convergence_counter=0		--add 1 every time 1 mean converges. once this == #clusters, convergence
		local Rijnumber=0			--#of times we update Rij
		randwt=torch.rand(mogm.k)
		wtsum=torch.sum(randwt)
		r=randwt/wtsum

		while (convergence_counter < mogm.k) do
--			mogm:likelihood(x)
			mogm:updateRij(x)
--			print("Rij updation number ")
			Rijnumber=Rijnumber+1
--			print(Rijnumber)
			convergence_counter=0
			for i=1,mogm.k do					---for every cluster
--				print("enter outer loop")
				local index=0
				for j=1,mogm.m do				---check in every sample
--					print("enter inner loop")
					a,b=torch.max(Rij[j],1)
--[[					print("value of i ")
					print(i)
					print("value of b")
					print(b[1])
]]					if(b[1]==i) then
--						print("match")
--						io.read()
						index=index+1
						clusterwise_data[index]=x[j]
					end
				end
--				print("inner loop finishes single iteration")
--				io.read()
--				r=torch.Tensor(index):fill(1)
				p=0.001
--				if index>0 then
					convergence_counter = convergence_counter + mogm[i]:learn(clusterwise_data,r[i],index,p)
--				end
			end
--			mogm:likelihood(x)			
			mogm:updatewt()
		end

		print("Max -ve likelihood function value after Gaussian converges")
		mogm:likelihood(x)
		mogm:updateRij(x)
		print("\nConvergence achieved for Gaussians, reconstructing image")

--		io.read()
		return Rij
	end

	function mogm:updatewt()
--           	mogm:likelihood(x)

		local den = 0
--[[		for i=1,mogm.m do
			for j=1,mogm.k do
				den = den + Rij[i][j]
			end
		end
]]
		den=torch.sum(Rij)
		local num = 0
		num=torch.sum(Rij,1)
--		for j=1,mogm.k do
			r=num/den
--		end
--		print(r)
--		print("this is weight")
--		print(r)
--		io.read()
		r:resize(mogm.k)
	end

	function mogm:updateRij(x)
		local v=torch.Tensor(mogm.k):fill(0)
		for i=1,mogm.m do
			v=mogm:g(x[i])
			denominator=torch.dot(v,r)
--			Rij[i]=(v*r)/denominator
			
--			print(denominator)
--			print("thats the denominator for sample i=")
--			print(i)
--			io.read()
			for j=1,mogm.k do
				Rij[i][j]=(v[j]*r[j])/denominator
				
			end
--			Rij[i]=
		end
	end

  	function mogm:likelihood(x)
		local temp = 0		    
  	         local v=torch.Tensor(mogm.k):fill(0)
		for i=1,mogm.m do
			v=mogm:g(x[i])

--			Rij[i]=(v*r)/denominator
			
--			print(denominator)
--			print("thats the denominator for sample i=")
--			print(i)
--			io.read()
			for j=1,mogm.k do
				likVal[i][1]=likVal[i][1] + (v[j]*r[j])
	
			end
--[[
			if(likVal[i][1]==0) then
				print("le le, ZERO ho gaya")
				io.read()
			end
]]				
				likVal[i][1]= -1*(torch.log(likVal[i][1]))
				temp=temp + likVal[i][1]

--	
		end
		print(temp)	
         end



	function mogm:f(x)
		local cluster_prob=mogm:g(x)
		a,max_cluster_prob=torch.max(cluster_prob,1)
		return mogm[max_cluster_prob[1]]
	end

	function mogm:g(x)
--                local min =  99999
                local cluster_prob = torch.Tensor(mogm.k):fill(0)
                for i=1,mogm.k do
                        cluster_prob[i]=mogm[i]:eval(x)
--[[[                        if edist<min then
                                min = edist
                                cluster_number = i
                        end
]]
                end
                return cluster_prob
	end

	return mogm
end