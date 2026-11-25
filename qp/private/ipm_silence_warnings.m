function ws = ipm_silence_warnings()
% Turn off near-singularity warnings (near convergence lam./t spans many decades); returns the
% previous states, for warning(ws)
    ws = warning('off', 'MATLAB:nearlySingularMatrix');
    ws(2) = warning('off', 'MATLAB:singularMatrix');
end
