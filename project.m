%% PageRank & Community Detection on the SNAP email-Eu-core network
% ===============================================================
% Author: Michelle Lawson & Adriana Soldat  
% Course: Linear Algebra – Final Project
%
% *Loads the dataset, builds a weighted digraph, computes PageRank, explores
% Laplacian communities, and offers live-update simulation tools.*
%
% Place `email-Eu-core.txt` and `email-Eu-core-department-labels.txt` in the
% same folder, then run:
%    project
%
% (or run sections in MATLAB for interactive experiments)

%% Initialize
clear all;
close all;

% File paths - using absolute paths to ensure files are found
EDGE_FILE = '/Users/michelle/Documents/GitHub/linear-algebra-final/email-Eu-core.txt';
LABEL_FILE = '/Users/michelle/Documents/GitHub/linear-algebra-final/email-Eu-core-department-labels.txt';

%% Data loading
function [G, weights, labels] = load_dataset(edge_path, label_path)
    % Return a weighted directed graph + department labels
    % G: adjacency matrix
    % weights: edge weights matrix
    % labels: node department labels
    
    if ~exist(edge_path, 'file') || ~exist(label_path, 'file')
        error('Dataset files missing – download from SNAP and retry.');
    end
    
    % Read edge data
    edgeData = dlmread(edge_path);
    
    % Get max node ID to determine matrix size
    maxNode = max(max(edgeData));
    
    % Create empty adjacency and weight matrices
    G = sparse(maxNode+1, maxNode+1);
    weights = sparse(maxNode+1, maxNode+1);
    
    % Fill adjacency and weight matrices
    for i = 1:size(edgeData, 1)
        u = edgeData(i, 1) + 1; % +1 because MATLAB is 1-indexed
        v = edgeData(i, 2) + 1;
        G(u, v) = 1;
        weights(u, v) = weights(u, v) + 1;
    end
    
    % Read label data
    labelData = dlmread(label_path);
    labels = zeros(maxNode+1, 1);
    
    % Fill label vector
    for i = 1:size(labelData, 1)
        node = labelData(i, 1) + 1; % +1 because MATLAB is 1-indexed
        dept = labelData(i, 2);
        labels(node) = dept;
    end
end

%% PageRank implementation
function pr = pagerank(G, weights, alpha, tol, max_iter)
    % Compute PageRank scores for nodes in graph G
    % G: adjacency matrix
    % weights: edge weights matrix
    % alpha: damping factor
    % tol: convergence tolerance
    % max_iter: maximum number of iterations
    % pr: PageRank vector
    
    if nargin < 3
        alpha = 0.85;
    end
    if nargin < 4
        tol = 1e-6;
    end
    if nargin < 5
        max_iter = 100;
    end
    
    n = size(G, 1);
    
    % Initialize PageRank vector
    pr = ones(n, 1) / n;
    
    % Get weighted out-degree for each node
    out_degree = sum(weights, 2);
    
    % Handle dangling nodes (no outgoing edges)
    dangling = (out_degree == 0);
    
    for iter = 1:max_iter
        % Store previous PR for convergence check
        prev_pr = pr;
        
        % Create transition matrix with weights
        M = weights ./ repmat(out_degree, 1, n);
        M(isnan(M)) = 0; % Fix division by zero for dangling nodes
        
        % Random teleportation contribution
        rand_contribution = (alpha * sum(pr(dangling)) / n) + ((1 - alpha) / n);
        
        % Matrix-vector multiplication for new PageRank
        pr = alpha * M' * pr + rand_contribution * ones(n, 1);
        
        % Normalize for numerical stability
        pr = pr / sum(pr);
        
        % Check for convergence
        if norm(pr - prev_pr, 1) < tol
            break;
        end
    end
end

%% Display top-k nodes
function display_topk(pr, k)
    % Display top k nodes by PageRank score
    if nargin < 2
        k = 10;
    end
    
    % Convert 1-indexed to 0-indexed for display
    pr_zero_indexed = [(0:length(pr)-1)', pr];
    
    % Skip zero entry (representing no node)
    pr_zero_indexed = pr_zero_indexed(2:end, :);
    
    % Sort by PageRank score
    [~, idx] = sort(pr_zero_indexed(:, 2), 'descend');
    top_k = pr_zero_indexed(idx(1:min(k, length(idx))), :);
    
    % Display results
    fprintf('Top-%d influential nodes:\n', k);
    fprintf('%-10s %-15s\n', 'node', 'PageRank score');
    fprintf('%-10d %-15.6f\n', top_k');
end

%% Laplacian zero-eigenvalue multiplicity
function dim = laplacian_zero_eigenspace_dim(G, weights)
    % Calculate dimension of zero-eigenvalue eigenspace of the Laplacian
    % G: adjacency matrix
    % weights: edge weights matrix
    
    % Make symmetric (undirected) for Laplacian
    H = max(G, G');
    W = max(weights, weights');
    
    % Compute weighted Laplacian
    degrees = sum(W, 2);
    D = diag(degrees);
    L = D - W;
    
    % Compute eigenvalues
    eigvals = eig(L);
    
    % Count eigenvalues close to zero
    dim = sum(abs(eigvals) < 1e-8);
end

%% Visualization
function plot_graph(G, weights, pr, title_str)
    % Plot graph with PageRank heatmap
    % G: adjacency matrix
    % weights: edge weights matrix
    % pr: PageRank vector
    % title_str: plot title
    
    if nargin < 4
        title_str = 'PageRank heatmap';
    end
    
    % Get nonzero elements from adjacency matrix to create edge list
    [src, dst] = find(G);
    edges = [src, dst];
    
    % Extract weights for these edges
    edge_weights = zeros(size(edges, 1), 1);
    for i = 1:length(src)
        edge_weights(i) = weights(src(i), dst(i));
    end
    
    % Create graph from edge list
    graph = digraph(edges(:,1), edges(:,2), edge_weights);
    
    % Normalize PR scores for coloring
    pr_norm = pr(1:max(max(src), max(dst)));
    pr_norm = pr_norm - min(pr_norm);
    if max(pr_norm) > 0
        pr_norm = pr_norm / max(pr_norm);
    end
    
    % Create figure
    figure('Position', [100, 100, 800, 650]);
    
    % Plot graph
    h = plot(graph, 'Layout', 'force');
    
    % Set node colors based on PR
    h.NodeCData = pr_norm;
    h.MarkerSize = 6;
    h.LineWidth = 0.5;
    colormap('cool');
    colorbar('Label', 'Relative PageRank score');
    
    % Style edges
    h.EdgeAlpha = 0.3;
    h.EdgeColor = [0.5 0.5 0.5];
    
    % Add title and remove axis
    title(title_str);
    axis off;
end

%% Simulation helpers
function weights = send_email(weights, sender, recipient, n)
    % Send n emails from sender to recipient
    % weights: edge weights matrix
    % sender, recipient: node indices
    % n: number of emails to send
    
    if nargin < 4
        n = 1;
    end
    
    % Update weight
    weights(sender, recipient) = weights(sender, recipient) + n;
end

function [weights, pr] = simulate_messages(G, weights, interactions, n, alpha)
    % Simulate message interactions and recompute PageRank
    % G: adjacency matrix
    % weights: edge weights matrix
    % interactions: cell array of [sender, recipient] pairs
    % n: number of emails per interaction
    % alpha: damping factor
    % weights: updated weights
    % pr: updated PageRank vector
    
    if nargin < 4
        n = 1;
    end
    if nargin < 5
        alpha = 0.85;
    end
    
    % Update graph with new interactions
    for i = 1:size(interactions, 1)
        sender = interactions(i, 1);
        recipient = interactions(i, 2);
        weights = send_email(weights, sender, recipient, n);
        G(sender, recipient) = 1;  % Ensure edge exists in adjacency matrix
    end
    
    % Recompute PageRank
    pr = pagerank(G, weights, alpha);
end

%% Main function
function main()
    % Main program execution
    
    % Using absolute file paths to ensure files are found
    EDGE_FILE = '/Users/michelle/Documents/GitHub/linear-algebra-final/email-Eu-core.txt';
    LABEL_FILE = '/Users/michelle/Documents/GitHub/linear-algebra-final/email-Eu-core-department-labels.txt';
    
    % Load dataset
    [G, weights, labels] = load_dataset(EDGE_FILE, LABEL_FILE);
    
    % Compute initial PageRank
    pr = pagerank(G, weights);
    
    % Display top nodes
    display_topk(pr);
    
    % Show Laplacian analysis
    fprintf('\nLaplacian zero-eigenvalue multiplicity (undirected view): %d\n', ...
            laplacian_zero_eigenspace_dim(G, weights));
    
    % Plot initial state
    plot_graph(G, weights, pr, 'Initial PageRank heatmap');
    
    % Simulate extra emails
    sender = 1;    % 0-indexed node becomes 1 in MATLAB
    recipient = 43; % 42-indexed node becomes 43 in MATLAB
    fprintf('\nSimulating 100 extra e-mails from %d → %d …\n', sender-1, recipient-1);
    
    [weights, pr2] = simulate_messages(G, weights, [sender, recipient], 100);
    
    % Plot updated state
    plot_graph(G, weights, pr2, 'After simulated messages');
end

%% Execute the main function
main();