% Copyright (c) 2020 Kosala Herath
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

% Kohonen's SOM
% Example 03: Triangle Input Distribution with 1D neuron array

clc;
clear all;
close all;

% parmaters

number_of_inputs=1000;
N=100; % N points

upper_bound_x=1;
lower_bound_x=-1;
upper_bound_m=0.1;
lower_bound_m=-0.1;

alpha_initial=0.2;
sigma_initial=0.005;
neighbour_radius_initial=N/2;

T=300; % number of iterations
t=1;

% initiate input and neural filed

points = zeros(number_of_inputs,2);
n = 0;

while (n < number_of_inputs)
    n = n + 1;
    x_i = 2*rand-1; % generate a number between -1 and 1
    y_i = 2*rand-1; % generate a number between -1 and 1
    if (y_i > -2 *abs(x_i) + 1)
       n = n - 1; % decrease the counter to try to generate one more point
    else % if the point is inside the triangle
       points(n,:) = [x_i y_i]; % add it to a list of points
    end
end

for j1=1:N
        m1(j1)=rand*(upper_bound_m-lower_bound_m)+lower_bound_m;
        m2(j1)=rand*(upper_bound_m-lower_bound_m)+lower_bound_m;
end

% plot the points generated
x1 = points(:,1);
x2 = points(:,2);

% initial figures

figure(1)
% input points in the x1,x2 map
plot(x1,x2,'ob')
hold on
% neural field points in the x1,x2 map
plot(m1,m2,'or')
plot(m1,m2,'r','linewidth',1)
% plot(w1',w2','r','linewidth',1)
hold off
title('t=0');
drawnow

pause(7);

% start traning

while (t<=T)
    
    % update parameters
    alpha=alpha_initial*(1-t/T);
    sigma=sigma_initial*(1-t/T);
    max_neighbour_radius=round(neighbour_radius_initial*(1-t/T));
    
    % loop over all the input values
    for i=1:number_of_inputs % took one input : a 2D vector
        
        % find minumum distance neural unit (winner)
        e_norm=(x1(i)-m1).^2+(x2(i)-m2).^2; % error distance for each neural node (output error matrix)
        minj1=1;
        min_norm=e_norm(minj1); % select fist element in matrix
        for j1=1:N
                if e_norm(j1)<min_norm
                    min_norm=e_norm(j1);
                    minj1=j1;
                end
        end
        % winner coordinates
        j1_c= minj1;
        
        % update the winning neuron
        e_factor = exp(-((j1_c-j1_c).^2)/2*sigma);
        m1(j1_c)=m1(j1_c) + alpha * (x1(i) - m1(j1_c));
        m2(j1_c)=m2(j1_c) + alpha * (x2(i) - m2(j1_c));
        
        % update the neighbour neurons
        for neighbour_radius=1:1:max_neighbour_radius
            jj1=j1_c - neighbour_radius;
            if (jj1>=1) % to stay in the matrix
                e_factor = exp(-((j1_c-jj1).^2)/2*sigma);
                m1(jj1)=m1(jj1) + alpha * e_factor * (x1(i)-m1(jj1));
                m2(jj1)=m2(jj1) + alpha * e_factor * (x2(i)-m2(jj1));
            end
            jj1=j1_c + neighbour_radius;
            if (jj1<=N) % to stay in the matrix
                e_factor = exp(-((j1_c-jj1).^2)/2*sigma);
                m1(jj1)=m1(jj1) + alpha * e_factor * (x1(i)-m1(jj1));
                m2(jj1)=m2(jj1) + alpha * e_factor * (x2(i)-m2(jj1));
            end
        end        
    end
    t=t+1;
    figure(1)
    plot(x1,x2,'ob')
    hold on
    plot(m1,m2,'or')
    plot(m1,m2,'r','linewidth',1)
    hold off
    title(['t=' num2str(t)]);
    drawnow
    
end