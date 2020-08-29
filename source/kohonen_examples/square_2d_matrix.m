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
% Example 01: Square Input Distribution

clc;
clear all;
close all;

% parmaters

number_of_inputs=1000;
N=10; % N^2 points

upper_bound_x=1;
lower_bound_x=-1;
upper_bound_m=0.1;
lower_bound_m=-0.1;

alpha_initial=0.1;
sigma_initial=0.05;
neighbour_radius_initial=N/2;

T=300; % number of iterations
t=1;

% initiate input and neural filed

for i=1:number_of_inputs
    x1(i)=rand*(upper_bound_x-lower_bound_x)+lower_bound_x;
    x2(i)=rand*(upper_bound_x-lower_bound_x)+lower_bound_x;
end
for j1=1:N
    for j2=1:N
        w1(j1,j2)=rand*(upper_bound_m-lower_bound_m)+lower_bound_m;
        w2(j1,j2)=rand*(upper_bound_m-lower_bound_m)+lower_bound_m;
    end
end

% initial figures

figure(1)
% input points in the x1,x2 map
plot(x1,x2,'ob')
hold on
% neural field points in the x1,x2 map
plot(w1,w2,'or')
plot(w1,w2,'r','linewidth',1)
plot(w1',w2','r','linewidth',1)
hold off
title('t=0');
drawnow

% start traning

while (t<=T)
    
    % update parameters
    alpha=alpha_initial*(1-t/T);
    sigma=sigma_initial*(1-t/T);
    max_neighbour_radius=round(neighbour_radius_initial*(1-t/T));
    
    % loop over all the input values
    for i=1:number_of_inputs % took one input : a 2D vector
        
        % find minumum distance neural unit (winner)
        e_norm=(x1(i)-w1).^2+(x2(i)-w2).^2; % error distance for each neural node (output error matrix)
        minj1=1;minj2=1;
        min_norm=e_norm(minj1,minj2); % select fist element in matrix
        for j1=1:N
            for j2=1:N
                if e_norm(j1,j2)<min_norm
                    min_norm=e_norm(j1,j2);
                    minj1=j1;
                    minj2=j2;
                end
            end
        end
        % winner coordinates
        j1_c= minj1;
        j2_c= minj2;
        
        % update the winning neuron
        e_factor = exp(-((j1_c-j1_c).^2+(j2_c-j1_c).^2)/2*sigma);
        w1(j1_c,j2_c)=w1(j1_c,j2_c) + alpha * (x1(i) - w1(j1_c,j2_c));
        w2(j1_c,j2_c)=w2(j1_c,j2_c) + alpha * (x2(i) - w2(j1_c,j2_c));
        
        % update the neighbour neurons
        for neighbour_radius=1:1:max_neighbour_radius
            jj1=j1_c - neighbour_radius;
            jj2=j2_c;
            if (jj1>=1) % to stay in the matrix
                e_factor = exp(-((j1_c-jj1).^2+(j2_c-jj2).^2)/2*sigma);
                w1(jj1,jj2)=w1(jj1,jj2) + alpha * e_factor * (x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2) + alpha * e_factor * (x2(i)-w2(jj1,jj2));
            end
            jj1=j1_c + neighbour_radius;
            jj2=j2_c;
            if (jj1<=N) % to stay in the matrix
                e_factor = exp(-((j1_c-jj1).^2+(j2_c-jj2).^2)/2*sigma);
                w1(jj1,jj2)=w1(jj1,jj2) + alpha * e_factor * (x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2) + alpha * e_factor * (x2(i)-w2(jj1,jj2));
            end
            jj1=j1_c;
            jj2=j2_c - neighbour_radius;
            if (jj2>=1) % to stay in the matrix
                e_factor = exp(-((j1_c-jj1).^2+(j2_c-jj2).^2)/2*sigma);
                w1(jj1,jj2)=w1(jj1,jj2) + alpha * e_factor * (x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2) + alpha * e_factor * (x2(i)-w2(jj1,jj2));
            end
            jj1=j1_c;
            jj2=j2_c + neighbour_radius;
            if (jj2<=N) % to stay in the matrix
                e_factor = exp(-((j1_c-jj1).^2+(j2_c-jj2).^2)/2*sigma);
                w1(jj1,jj2)=w1(jj1,jj2) + alpha * e_factor * (x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2) + alpha * e_factor * (x2(i)-w2(jj1,jj2));
            end
        end        
    end
    t=t+1;
    figure(1)
    plot(x1,x2,'ob')
    hold on
    plot(w1,w2,'or')
    plot(w1,w2,'r','linewidth',1)
    plot(w1',w2','r','linewidth',1)
    hold off
    title(['t=' num2str(t)]);
    drawnow
    
end
