function [ theta, B ] = GeoNMF( A, K, epsilon, q )
%
% GeoNMF of MMSB inference
% Input: A: Adjacency matrix
%        K: number of clusters
%        q: quantile for choosing an almost pure node in each iteration for robustness, default as 0.95
%        epsilon: maximum error for choosing pure nodes at once, default as 0.25
% Output: theta: node-community matrix, theta_{ij} is the probability node i is in community j
%         B: community-community matrix, B_{ij} is the probability there is
%         an edge between a node in community i and a node in community j

% Author: Xueyu Mao 
% Email: maoxueyu@gmail.com
% Last Update: Mar 20, 2017
%
% Reference: On Mixed Memberships and Symmetric Nonnegative Matrix
% Factorizations, Xueyu Mao, Purnamrita Sarkar, and Deepayan Chakrabart, in
% ICML, 2017. ArXiv: 1607.00084

    if nargin <= 3
        q = 0.95;
        if nargin <=2
            epsilon = 0.25;
        end
    end

    epsilon_unit = 0.025;
    q_unit = 0.01;
    epsilon_tre = 0.95;

    N = size(A,1);

    [ Vo, Eo ] = eigs( A, K,'la');

    d = sum(A,2);
    % For relative sparse matrix using a weaker condition number threshold
    if mean(d)<250
        cond_tre = 2;
        q_tre = 0.7;
    else
        cond_tre = 1.5;
        q_tre = 0.8;
    end

    d_ph = sparse(1:N,1:N,sqrt(d));
    d_ng = sparse(1:N,1:N,1./sqrt(d));
    % Squaret root of Laplacian
    Xo = d_ng * Vo * sqrtm(Eo);

    a = sum(Xo.^2,2);

    update_epsikon_flag = 1;
    at_least_once = 0;
    while update_epsikon_flag && epsilon<=epsilon_tre || at_least_once==0 %0.95
        conn_old = cond_tre;
        S = find(a>(1-epsilon)*max(a));
        % Find pure nodes set that has pure nodes of all communities
        while length(S)<K && (epsilon <epsilon_tre)
            epsilon = epsilon + epsilon_unit;
            S = find(a>(1-epsilon)*max(a));
        end

        X_4_clstr = Xo(S,:);
        % K-means clustring to cluster the pure nodes set 
        sumd_min = Inf;
        for ii = 1:20
            [idx_tmp,~,sumd] = kmeans(X_4_clstr,K);
            if sum(sumd) < sumd_min
                sum_min = sum(sumd);
                idx = idx_tmp;
            end
        end
        
        B = zeros(K,K);
        X_pure = [];
        pure = [];
        pure_K = [];
        % Find an almost pure node in each iteration
        for i = 1:K
            S_i = find(idx==i);

            pure_tmp = S(S_i);
            X_pure_tmp = X_4_clstr(S_i,:);
            X_beta = d_ph(pure_tmp,pure_tmp) * X_pure_tmp;

            beta_tmp = sum(X_beta.^2,2);

            S_q = find(beta_tmp>=quantile(beta_tmp,q));
            while isempty(S_q) && q > q_unit
                q = q - q_unit;
                if q == q_unit
                    S_q = 1:length(beta_tmp);
                else
                    S_q = find(beta_tmp>=quantile(beta_tmp,q));
                end
            end

            [~, I_q] = sort(beta_tmp(S_q),'descend');
            pure_loc = S_i(S_q(I_q(end)));
            % Estimate B(i,i)
            B(i,i) = mean(beta_tmp);

            pure = [pure; pure_loc];
            pure_K = [pure_K S_q(I_q(end))];

            X_pure = [X_pure; X_4_clstr(pure_loc,:)];

        end

        pure = S(pure);

        if cond(Xo(pure,:)) < cond_tre
            update_epsikon_flag = 0;
        elseif conn_old > cond_tre && conn_old < 50
            break;
        elseif epsilon <= 0.4
            epsilon = epsilon + epsilon_unit;
        else
        if q > q_unit
                q = q - q_unit;
        else
        break
        end
            epsilon = epsilon + epsilon_unit;
            if q <= q_tre % 0.8
                break
            end
        end

        at_least_once = 1;

    end     
    % Estimate theta
    theta = d_ph * Xo * inv(Xo(pure,:)) * d_ng(pure,pure); 
    theta(theta<0) = 0;
    [ theta ] = normalize_row_l1( theta );
    
    % ALternative method for estimating B
%     [ B_est ] = est_B_from_theta_A( theta, A );

end

% Function for normalizing rows to theta to have unit l1 norm
function [ theta_l1 ] = normalize_row_l1( theta )

    theta_l1 = bsxfun(@times, theta, 1./(max(sum(theta, 2), eps)));

end

% ALternative method for estimating B
function [ B_est ] = est_B_from_theta_A( theta_est, A )

    B_est=inv(theta_est'*theta_est)*theta_est'*A*theta_est*inv(theta_est'*theta_est);

    B_est(B_est<0) = 0;
    B_est(B_est>1) = 1;

    % B_est=diag(diag(B_est));

end


