%% Copyright 2020 Alexander Liniger

%% Licensed under the Apache License, Version 2.0 (the "License");
%% you may not use this file except in compliance with the License.
%% You may obtain a copy of the License at

%%     http://www.apache.org/licenses/LICENSE-2.0

%% Unless required by applicable law or agreed to in writing, software
%% distributed under the License is distributed on an "AS IS" BASIS,
%% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%% See the License for the specific language governing permissions and
%% limitations under the License.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/

clear
close all
clc
%%
fileID = fopen('Disc_Test-285.bin');
KN_CM = fread(fileID,'bool');
r_max = 285
a_max = 1.6

% NGD = 201;
% NGMU = 161;
% NGVX = 270;
% 
NGD = 101;
NGMU = 81;
NGVX = 135;


KN = zeros(NGD+2,NGMU+2,NGVX+2);

for k = 1:NGVX
    for j = 1:NGMU
        for i = 1:NGD
            KN(i+1,j+1,k+1) = KN_CM(k + (j-1)*NGVX + (i-1)*NGVX*NGMU);
        end
    end
end

%%
d_start = -0.3415;
d_end = 0.3415;
d_diff = (d_end - d_start)/(NGD-1);
d_vec = (d_start - d_diff):d_diff:(d_end + d_diff);

mu_start = -0.2;
mu_end = 0.2;
mu_diff = (mu_end - mu_start)/(NGMU-1);
mu_vec = (mu_start - mu_diff):mu_diff:(mu_end + mu_diff);

v_start = 0;
v_end = min(sqrt(kappa_max*a_max),35);
v_diff = (v_end - v_start)/(NGVX-1);
v_vec = (v_start - v_diff):v_diff:(v_end + v_diff);

[d1,mu1,vx1] = meshgrid(mu_vec,d_vec,v_vec);
%%

figure(3)
v = KN;
p = patch(isosurface(d1,mu1,vx1,v,0));
isonormals(d1,mu1,vx1,v,p)
set(p,'FaceColor','y','EdgeColor','none');
daspect([1,1,15])
view(3); axis tight
camlight 
% lighting gouraud
lighting phong;
% axis on
grid on
xlim([-.25,0.25])
ylim([-.4,0.4])
zlim([0,v_end+1])
xlabel('\mu [rad]','FontSize',12)
ylabel('d [m]','FontSize',12)
zlabel('v [m/s]','FontSize',12)


