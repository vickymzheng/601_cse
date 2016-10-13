
%% get frequent itemset of size 1
% ups: 2*gene_num-1: 1,3,5,7.. downs: 2*gene_num: 2:4:6:8
% 4 disease 201,202,203,204
clear all
load('dataset.mat');
support = 0.3;
fi = [];    % current round of frequent itemset of size k
fis = {};   % cell array, each cell is the frequent itemset of size k

% k = 1
for i = 1:100
    % UP
    up_num = length( find(gene(:,i) == 1));
    down_num = length( find(gene(:,i) == 0));
    if up_num >= support * sample_num
        fi = [fi; 2*i-1];
    end
    
     if down_num >= support * sample_num
        fi = [fi; 2*i];
    end
end

for i = 1:4
    d_num = length( find(gene(:,101) == i));
    if d_num >= support * sample_num
        fi = [fi; 200+i];
    end
    
end

fis{1} = fi;

fitem_num = size(fis{1},1); % the number of freqent itemset of largest k
k =1;
flag = 1;

while flag == 1
    k = k + 1; % first round k =2, need to get fs with k =2
    last_fs = fis{k-1};
    fitem_num = size(fis{1},1);
    fi = [];
    
    for is1 = 1:fitem_num-1
        for is2 = is1+1 : fitem_num
            union_set = union( last_fs(is1,:) , last_fs(is2,:) );
            if (size( union_set,2) == k) 
                if union_set_frq(union_set,last_fs,gene, support) == 1 % dont contain infreqent subsets, support > threshold_support
                    fi = [fi ; union_set];
                end
            end
        end        
    end
    
    if size(fi,1) == 1 || isempty(fi) == 1
        flag = 0;
    end
    
    if isempty(fi) == 0
        fis{k} = fi;
    end
    
    
end
