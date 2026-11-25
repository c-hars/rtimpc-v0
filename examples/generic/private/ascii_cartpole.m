function ascii_cartpole(X, U, Ts, opts)
% ASCII_CARTPOLE  Stop-motion replay of a cart-pole trajectory in the terminal.
%
%   ascii_cartpole(X, U, Ts)
%     X  : states, one column per sample: [x; xdot; theta; thetadot], theta = 0 upright
%          (cart position in row 1; the pole angle in row 3 unless Angles is given)
%     U  : 1 x K forces
%     Ts : sample time [s]
%
%   Options (Name=Value):
%     TrackLimit : |x| limit of the cart, drawn as end stops (default: none)
%     PoleLength : drawn pole length [m] (default 0.5); a vector draws a chain of links
%     Angles     : absolute link angles, one row per link (default: X(3,:)), theta = 0 upright
%     Speed      : playback rate, 1 = real time (default 1)
%     Frames     : frame indices to print one below another instead of animating
%                  (for logs, or a stop-motion strip), e.g. 1:10:size(X,2)
%     Width      : drawing width in characters (default 60); the height follows from it
%     Rows       : terminal height in lines, used to draw the frames at the bottom of the
%                  window (default: from get(0, 'CommandWindowSize'); 0 = top-aligned)
%
%   Frames are redrawn with clc, which works in the MATLAB desktop, a terminal (-nodesktop)
%   and the VS Code MATLAB extension (whose terminal swallows ANSI escape codes). Under
%   -batch, clc does nothing and the frames are appended.

    arguments
        X double
        U (1,:) double
        Ts (1,1) double
        opts.TrackLimit (1,1) double = Inf
        opts.PoleLength (1,:) double = 0.5
        opts.Angles double = []
        opts.Speed (1,1) double = 1
        opts.Frames double = []
        opts.Width (1,1) double = 60
        opts.Rows (1,1) double = NaN
    end

    Ls = opts.PoleLength;  l = sum(Ls);                  % links; l: reach of the chain
    Th = opts.Angles;  if isempty(Th), Th = X(3,:); end
    Th = atan2(sin(Th), cos(Th));
    assert(size(Th, 1) == numel(Ls), 'ascii_cartpole: one row of Angles per PoleLength')
    Fmax = max([abs(U), eps]);
    cart_half = 0.12;                                    % [m] drawn half-width of the cart

    % Geometry: physical metres -> character cells. Terminal cells are about twice as tall as
    % they are wide, so 1 m is twice as many columns as rows (keeps the pole's angle true).
    reach = max(abs(X(1,:)));
    if isfinite(opts.TrackLimit), reach = max(reach, opts.TrackLimit); end
    span = reach + max(l, cart_half) + 0.1;              % half-width of the drawn window [m]
    W  = opts.Width;
    sx = (W - 1) / (2*span);                             % columns per metre
    sy = sx / 2;                                         % rows per metre
    up = ceil(l*sy);                                     % rows above the pivot
    p  = up + 1;                                         % pivot (cart) row
    H  = p + max(ceil(l*sy), 2) + 1;                     % hanging pole (rail, force arrow), status
    col = @(x) min(max(round((x + span)*sx) + 1, 1), W);
    row = @(y) min(max(p - round(y*sy), 1), H);

    render = @(k) frame(k);
    K = size(X, 2);

    if ~isempty(opts.Frames)                             % static strip
        for k = opts.Frames(:)'
            fprintf('%s\n\n', strjoin(cellstr(render(k)), newline));
        end
        return
    end

    % Bottom-align: pad above the frame so it sits on the last lines of the window
    rows = opts.Rows;
    if isnan(rows)
        sz = get(0, 'CommandWindowSize');
        rows = sz(2);
    end
    pad = repmat(newline, 1, max(rows - H - 1, 0));     % -1: the prompt line after the frame

    t0 = tic;
    for k = 1:K
        G = render(k);
        clc; fprintf('%s%s\n', pad, strjoin(cellstr(G), newline));
        % Hold real-time pacing, and always yield: without a pause (or drawnow) the MATLAB
        % desktop / IDE command windows do not repaint until the loop ends
        lag = (k-1)*Ts/opts.Speed - toc(t0);
        pause(max(lag, 0.01));
    end


    function G = frame(k)
        x  = X(1,k);
        G  = repmat(' ', H, W);

        % Rail with end stops
        G(p+1, :) = '-';
        if isfinite(opts.TrackLimit)
            for s = [-1 1]
                c = col(s*(opts.TrackLimit + cart_half));
                G(max(p-1, 1):p+1, c) = '#';
            end
        end

        % Links: sample each segment joint -> tip; the character follows the link's angle
        px = x;  py = 0;  tips = zeros(2, numel(Ls));
        for j = 1:numel(Ls)
            th = Th(j,k);  lj = Ls(j);
            b = mod(th, pi);                             % undirected angle from vertical
            if b < pi/8 || b >= 7*pi/8, ch = '|';
            elseif b < 3*pi/8,          ch = '/';
            elseif b < 5*pi/8,          ch = '-';
            else,                       ch = '\';
            end
            n = ceil(3 * lj * sx) + 2;
            for a = linspace(0, 1, n)
                G(row(py + a*lj*cos(th)), col(px + a*lj*sin(th))) = ch;
            end
            px = px + lj*sin(th);  py = py + lj*cos(th);
            tips(:, j) = [px; py];
        end
        for j = 1:numel(Ls)                              % joints ('o') and tip ('O') over the links
            G(row(tips(2,j)), col(tips(1,j))) = char('O' * (j == numel(Ls)) + 'o' * (j < numel(Ls)));
        end

        % Cart body on the pivot row, wheels on the rail, pivot on top
        c0 = col(x - cart_half); c1 = col(x + cart_half);
        G(p, c0:c1) = '=';  G(p, c0) = '[';  G(p, c1) = ']';
        G(p+1, [c0+1, c1-1]) = 'o';
        G(p, col(x)) = '+';

        % Force arrow under the rail, length ~ |F|
        if k <= numel(U)
            F = U(k);
            len = round(10 * abs(F) / Fmax);
            if len > 0
                cs = col(x) + sign(F)*(1:len);
                cs = cs(cs >= 1 & cs <= W);
                G(p+2, cs) = '=';
                if ~isempty(cs), G(p+2, cs(end)) = char('>' * (F > 0) + '<' * (F < 0)); end
            end
        else
            F = NaN;
        end

        % Status line
        status = sprintf(' t = %4.2f s  x = %+5.2f m  theta = %s deg  F = %+5.1f N', ...
            (k-1)*Ts, x, strjoin(compose('%+6.1f', rad2deg(Th(:,k))), ','), F);
        G(H, 1:min(W, numel(status))) = status(1:min(W, numel(status)));
    end
end
