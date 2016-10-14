function [ freqent ] = union_set_frq(union_set,last_fs,gene, support)
%   freqent if union_set dont contain infreqent subsets, support > threshold_support
%   union_set is 1*k
%   last_fs is freq_is_num * (k-1)
%   freqent = 1 if union_set is freqent; otherwise, freqent = 0

k = size(union_set,2);
last_fs_num = size(last_fs,1);
sample_num = size(gene,1);

freqent = 1;
for i = 1:k
    tem_set = union_set;
    tem_set(i) = [];
    
    if ismember(tem_set,last_fs, 'rows') == 0
        freqent = 0;
        break;
    end   
end

if freqent == 1 % support check
    count = 0;
    for i = 1:sample_num
        if sample_contain(union_set,gene(i,:)) == 1
            count = count+1;
        end
        
        if count >= support * sample_num
                freqent = 1;
                break;
        end
    end
    
    if count < support * sample_num
        freqent = 0;
    end
    
    
    
end

end

