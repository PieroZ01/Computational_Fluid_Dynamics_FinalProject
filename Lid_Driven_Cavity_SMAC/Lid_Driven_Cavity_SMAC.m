clear
clc
close all

% Lid-Driven Cavity - SMAC method

% Define some parameters
% Side length of the square
L = 1;
% Number of cells in each direction
Nx = 50;
Ny = 50; % (we use a square grid, so Nx = Ny)
% Time step
dt = 0.0001;
% Viscosity
nu = 0.01;
% Reynolds number
Re = 100;
% Space steps
dx = L / (Nx - 1);
dy = L / (Ny - 1);
% Number of time steps
Nt = 60000;

% Create the space grid
x = linspace(0, L, Nx);
y = linspace(0, L, Ny);
[Y, X] = meshgrid(y, x);

% Build the matrices A and N for the Poisson equation (A*P=b)
% We need the matrix for the Jacobi method
N = Nx * Ny; % Dimension of the matrices
% Initialize the matrices
A = zeros(N, N);
D = zeros(N, N);
% Fill the matrices
for j=2:Nx-1
    for i=2:Nx-1
        index_ij  = Nx * (j-1) + i;
        index_top = Nx * (j-1) + i + Nx;
        index_bot = Nx * (j-1) + i - Nx;
        index_lef = Nx * (j-1) + i + 1;
        index_rig = Nx * (j-1) + i - 1;
        
        A(index_ij, index_ij)  = -4;    
        A(index_ij, index_rig) =  1;         % right coeff
        A(index_ij, index_lef) =  1;         % left coeff
        A(index_ij, index_bot) =  1;         % bottom coeff
        A(index_ij, index_lef) =  1;         % top coeff
    end
end
% Correct the boundary conditions
j=1;
for i=2:Nx-1
        index_ij  =  i;
        index_bot =  Nx + i;
        A(index_ij, index_ij)     = -4;    
        A(index_ij, index_ij + 1) =  1;         % right coeff
        A(index_ij, index_ij - 1) =  1;         % left coeff
        A(index_ij, index_bot)    =  2;         % bottom coeff - vertical derivative = 0
end
j=Nx;
for i=2:Nx-1
        index_ij  = Nx * (j-1) + i;
        index_top = Nx * (j-2) + i;
        A(index_ij, index_ij)     = -4;    
        A(index_ij, index_ij + 1) =  1;         % right coeff
        A(index_ij, index_ij - 1) =  1;         % left coeff
        A(index_ij, index_top)    =  2;         % top coeff - vertical derivative = 0
end
i = 1;
for j=2:Nx-1
        index_ij  = Nx * (j-1) + i;
        index_top = Nx * (j-2) + i;
        index_bot = Nx * j + i;
        A(index_ij, index_ij)     = -4;    
        A(index_ij, index_ij + 1) =  2;      % right coeff
        A(index_ij, index_top)    =  1;         % top coeff
        A(index_ij, index_bot)    =  1;         % bottom coeff
end
i = Nx;
for j=2:Nx-1
        index_ij  = Nx * (j-1) + i;
        index_top = Nx * (j-2) + i;
        index_bot = Nx * j + i;
        A(index_ij, index_ij)     = -4;    
        A(index_ij, index_ij - 1) =  2;      % left coeff
        A(index_ij, index_top)    =  1;         % top coeff
        A(index_ij, index_bot)    =  1;         % bottom coeff
end
% Corners
A(1, 1)      = 1;
A(N, N)  = 1;
A(Nx, Nx)    = 1;
A(Nx * (Nx-1) + 1, Nx * (Nx-1) + 1) = 1;
% Create the diagonal matrix D
for i=1:N
    D(i, i)   = A(i, i);
end
% Compute the inverse of N
N_inv = inv(D);

% Set the initial conditions
u = zeros(Nx, Ny, Nt);
v = zeros(Nx, Ny, Nt);
P = zeros(Nx, Ny, Nt);

% Check the stability conditions
Cu = dt * (max(abs(u), [], 'all') / dx + max(abs(v), [], 'all') / dy);
if Cu > 1
    % Throw an error
    error('Courant number is greater than 1');
end
% Diffusion number
Beta_x = dt / (dx * dx * Re);
Beta_y = dt / (dy * dy * Re);
if Beta_x > 0.25 || Beta_y > 0.25
    % Throw an error
    error('Diffusion number is greater than 0.25');
end

% Initialize the intermediate velocity field
u_star = zeros(Nx, Ny);
v_star = zeros(Nx, Ny);

% Define some auxiliary useful variables

source_u = zeros(Nx, Ny); % source term for the Poisson equation along x
source_v = zeros(Nx, Ny); % source term for the Poisson equation along y

source_vector = zeros(N, 1); % source term vector for the Poisson equation
P_vector = zeros(N, 1); % pressure field vector for the Poisson equation

grad_Px = zeros(Nx, Ny); % gradient of the pressure field along x
grad_Py = zeros(Nx, Ny); % gradient of the pressure field along y

nitermax = 300; % maximum number of iterations for the Jacobi method
eps = 0.000001; % tolerance for the Jacobi method

u_plot = zeros(Nt, 1); % variable for checking the state of the flow (if fully developed)

% Main time loop to solve the Navier-Stokes equations for the lid-driven cavity problem using the S-MAC method
for t = 2:Nt
    
    % PREDICTOR step to obtain the intermediate velocity field
    % We solve the advection-diffusion equations
    for i = 2:Nx-1
        for j = 2:Ny-1
            adv_ux = (u(i+1, j, t-1) * u(i+1, j, t-1) - u(i-1, j, t-1) * u(i-1, j, t-1)) / (2 * dx); % advection term
            adv_uy = (u(i, j+1, t-1) * v(i, j+1, t-1) - u(i, j-1, t-1) * v(i, j-1, t-1)) / (2 * dy); % advection term
            diff_u = (u(i+1, j, t-1) - 2 * u(i, j, t-1) + u(i-1, j, t-1)) / (dx * dx) + (u(i, j+1, t-1) - 2 * u(i, j, t-1) + u(i, j-1, t-1)) / (dy * dy); % diffusion term
            u_star(i, j) = u(i, j, t-1) - dt * (adv_ux + adv_uy) + dt * nu * diff_u; % intermediate velocity field u_star
            adv_vx = (v(i+1, j, t-1) * u(i+1, j, t-1) - v(i-1, j, t-1) * u(i-1, j, t-1)) / (2 * dx); % advection term
            adv_vy = (v(i, j+1, t-1) * v(i, j+1, t-1) - v(i, j-1, t-1) * v(i, j-1, t-1)) / (2 * dy); % advection term
            diff_v = (v(i+1, j, t-1) - 2 * v(i, j, t-1) + v(i-1, j, t-1)) / (dx * dx) + (v(i, j+1, t-1) - 2 * v(i, j, t-1) + v(i, j-1, t-1)) / (dy * dy); % diffusion term
            v_star(i, j) = v(i, j, t-1) - dt * (adv_vx + adv_vy) + dt * nu * diff_v; % intermediate velocity field v_star
        end
    end
    
    % Compute the source term for the Poisson equation
    for i = 2:Nx-1
        for j = 2:Ny-1
            source_u(i, j) = (u_star(i+1, j) - u_star(i-1, j)) / (2 * dx); % source term along x
            source_v(i, j) = (v_star(i, j+1) - v_star(i, j-1)) / (2 * dy); % source term along y
        end
    end
    
    % Apply the boundary conditions for the source term
    source_u(1, :) = 0;
    source_u(Nx, :) = 0;
    source_u(:, 1) = 0;
    source_u(:, Ny) = 0;
    source_v(1, :) = 0;
    source_v(Nx, :) = 0;
    source_v(:, 1) = 0;
    source_v(:, Ny) = 0;
    
    % Fill the source term vector (also the pressure field has to be reshaped as a 1D vector)
    count = 1;
    for i = 1:Nx
        for j = 1:Ny
            source_vector(count) = ((source_u(i, j) + source_v(i, j)) / dt) * dx * dx; % (Assuming dx = dy)
            P_vector(count) = P(i, j, t-1);
            count = count + 1;
        end
    end
    
    b = source_vector; % Rename the source term vector
    xk = P_vector; % The initial guess for the pressure field vector is the one at the previous time step
    
    % Jacobi method to solve the Poisson equation and obtain the pressure field
    count_iter = 1; % iteration counter
    res_mod = 1; % residual
    while count_iter < nitermax & res_mod > eps
        res = A * xk - b;
        xk_new = xk - N_inv * res;
        res_mod = norm(A * xk_new - b); % compute the residual magnitude
        xk = xk_new;
        count_iter = count_iter + 1;
    end
    
    % Print the number of iterations
    if mod(t, 200)==0
        fprintf('Time step: %d\n', t);
        fprintf('Number of iterations: %d\n', count_iter);
    end
    
    % Reshape the pressure field vector as a 2D matrix
    count = 1;
    for i = 1:Nx
        for j = 1:Ny
            P(i, j, t) = xk_new(count);
            count = count + 1;
        end
    end
    
    % Apply the boundary conditions for the pressure field
    P(:, Nx, t) = P(:, Nx-1, t);
    P(1, :, t) = P(2, :, t);
    P(Nx, :, t) = P(Nx-1, :, t);
    P(:, 1, t) = P(:, 2, t);
    
    % CORRECTOR step to obtain the velocity field
    for i = 2:Nx-1
        for j = 2:Ny-1
            grad_Px(i, j) = (P(i+1, j, t) - P(i-1, j, t)) / (2*dx); % gradient of the pressure field along x
            grad_Py(i, j) = (P(i, j+1, t) - P(i, j-1, t)) / (2*dy); % gradient of the pressure field along y
        end
    end
    
    % Compute the velocity field at the next time step using the gradient of the pressure field
    u(:, :, t) = u_star(:, :) - dt * grad_Px;
    v(:, :, t) = v_star(:, :) - dt * grad_Py;
    
    % Apply the boundary conditions for the velocity field
    u(1, :, t) = 0;
    v(1, :, t) = 0;
    u(Nx, :, t) = 0;
    v(Nx, :, t) = 0;
    u(:, 1, t) = 0;
    v(:, 1, t) = 0;
    u(:, Ny, t) = 1;
    v(:, Ny, t) = 0;

    % Store the central value of the u component of the velocity field for plotting
    % (Useful to see if the flow reached a steady state)
    u_plot(t, 1) = u(int32(Nx/2), int32(Ny/2), t);
    
    % Check the stability conditions
    Cu = dt * (max(abs(u), [], 'all') / dx + max(abs(v), [], 'all') / dy);
    if Cu > 1
        % Throw an error
        error('Courant number is greater than 1');
    end
    % Diffusion number
    Beta_x = dt / (dx * dx * Re);
    Beta_y = dt / (dy * dy * Re);
    if Beta_x > 0.25 || Beta_y > 0.25
        % Throw an error
        error('Diffusion number is greater than 0.25');
    end

end

figure;

% Plot the results
for t=1:100:Nt

    % Plot for u
    subplot(3, 1, 1);
    [C1, h1] = contourf(X, Y, u(:, :, t), 20); % Change with u, v, P
    set(h1, "Linecolor", "None")
    xlim([x(1), x(end)])
    ylim([y(1), y(end)])
    colorbar
    xlabel("x axis")
    ylabel("y axis")
    title('u Contour Plot');

    % Plot for v
    subplot(3, 1, 2); % Second subplot
    [C2, h2] = contourf(X, Y, v(:, :, t), 20);
    set(h2, 'LineColor', 'None');
    xlim([x(1), x(end)]);
    ylim([y(1), y(end)]);
    colorbar;
    xlabel('x axis');
    ylabel('y axis');
    title('v Contour Plot');

    % Plot for P
    subplot(3, 1, 3); % Third subplot
    [C3, h3] = contourf(X, Y, P(:, :, t), 20);
    set(h3, 'LineColor', 'None');
    xlim([x(1), x(end)]);
    ylim([y(1), y(end)]);
    colorbar;
    xlabel('x axis');
    ylabel('y axis');
    title('P Contour Plot');

    pause(0.00000000001)
end

%plot(u_plot)

figure;

subplot(1, 2, 1);
set(gca, 'Position', [0.1, 0.3, 0.35, 0.6]);
plot(y, u(int32(Nx/2), :, Nt), 'LineWidth', 2);
xlabel('y', 'FontSize', 14);
ylabel('u velocity', 'FontSize', 14);

subplot(1, 2, 2);
set(gca, 'Position', [0.55, 0.3, 0.35, 0.6]);
plot(x, u(:, int32(Ny/2), Nt), 'LineWidth', 2);
xlabel('x', 'FontSize', 14);
ylabel('u velocity', 'FontSize', 14);

clear n i j
