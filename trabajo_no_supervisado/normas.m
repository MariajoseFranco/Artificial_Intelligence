function norma = normas(distance, A, B, dataset)

[nA, mA] = size(A);
[nB, mB] = size(B);

if nA == nB && mA == mB
    if distance == 'euc'
        norma = sqrt(sum((A-B).^2));;
    
    elseif distance == 'mah'
        inv_cov = cov(dataset)^(-1);
        dist_cuad = (A-B)*inv_cov*transpose(A-B);
        %mahal_cuad = diag(dist_cuad);
        norma = sqrt(dist_cuad);

    elseif distance == 'man'
        norma = sum(abs(A-B));
        
    elseif distance == 'cos'
        norma_A = sum(A.^2);
        norma_B = sum(B.^2);
        norma = 1-(sum(A.*B)/(sqrt(norma_A*norma_B)));
    end
else
    disp('Error: A y B no tienen las mismas dimensiones')
end

end