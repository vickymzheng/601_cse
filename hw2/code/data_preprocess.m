[nrow, ncol] = size(geneexpression);
sample_num = nrow;
gene_num = ncol - 2;

gene_o = geneexpression;

for j = 2:(ncol-1)
    for i = 1: nrow
    
     s = strcat('G',num2str(j-1),'_',gene_o(i,j)   );
     geneexpression(i,j) = s;   
        
    end    
end

%% write to txt file
fileID = fopen('data.txt','w');
for i = 1: nrow
    for j = 1:ncol
        fprintf(fileID,'%s\t', cell2mat(geneexpression(i,j)));
    end
    fprintf(fileID,'\n');
end
fclose(fileID);

gene = [];
for i = 1:nrow
    for j = 2:101
        if(strcmp(gene_o(i,j), 'UP') )
            gene(i,j-1) = 1;
        end
        
        if(strcmp(gene_o(i,j), 'Down') )
            gene(i,j-1) = 0;
        end
    end
end

for i = 1:nrow
    if(strcmp(gene_o(i,102), 'ALL') )
            gene(i,101) = 1;
    end
    
    if(strcmp(gene_o(i,102), 'AML') )
            gene(i,101) = 2;
    end
    
    if(strcmp(gene_o(i,102), 'Breast Cancer') )
            gene(i,101) = 3;
    end
    
    if(strcmp(gene_o(i,102), 'Colon Cancer') )
            gene(i,101) = 4;
    end
    
    
end