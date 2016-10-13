function [ contain ] = sample_contain(union_set,this_sample)
%UNTITLED2 Summary of this function goes here
%   union_set is 1*k
%   this_sample is 1*item_nums, here is 1*101

k = size(union_set,2);
contain = 1;

for i = 1:k
    if union_set(i) <= 200
        index = floor ( ( union_set(i) + 1 ) / 2 );
        if this_sample(index) ~= mod ( union_set(i),2)
            contain = 0;
            break;
        end
        
    end
    
    if union_set(i) > 200
        if this_sample(101) ~= ( union_set(i)-200)
            contain = 0;
            break;
        end
    end
end

end

