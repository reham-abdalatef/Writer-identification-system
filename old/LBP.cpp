#include<bits/stdc++.h>
using namespace std;

extern "C" {
    double* solve(const char* p,int size){
    	freopen("o.txt","w",stdout);
    	string path(size);
		for(int i=0; i<size;i++)
			path[i] = p[2*i];

		// TODO
		// read image in path and call calcLBP function
		
		//return calcLBP(gray,n,m);
	}

	double* calcLBP(int** gray,int n,int m){
		int hist[256]={0};
		int threshold = 170;
		for(int i=1;i+1<n;i++)
			for(int j=1;j+1<m;j++){
				if(gray[i][j]>threshold) 
					continue;
				// 7 0 1
    			// 6 C 2
    			// 5 4 3
				int val = 0;
                val |= 1*(gray[i-1][j] <= gray[i][j]);
                val |= 2*(gray[i-1][j+1] <= gray[i][j]);
                val |= 4*(gray[i][j+1] <= gray[i][j]);
                val |= 8*(gray[i+1][j+1] <= gray[i][j]);
                val |= 16*(gray[i+1][j] <= gray[i][j]);
                val |= 32*(gray[i+1][j-1] <= gray[i][j]);
                val |= 64*(gray[i][j-1] <= gray[i][j]);
                val |= 128*(gray[i-1][j-1] <= gray[i][j]);
                ++hist[val];
			}			
		double sum = 0;
		for(int i=0;i<256;i++)
			sum+=hist[i];
		double * ret = new double[256];
		for(int i=0;i<256;i++)
			ret[i] = hist[i] / sum;
		return ret;
	}
}