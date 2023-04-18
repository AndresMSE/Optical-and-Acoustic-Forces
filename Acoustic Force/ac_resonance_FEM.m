% close("all")

%1st step create the model
model = createpde(1);
%2nd step create the geometry for the model
geom = geometryFromEdges(model,@squareg);
figure Name 'fig1-2'
pdegplot(model,'EdgeLabels','on');
axis equal 
title 'Geometría de estructura'
xlim([-1.1 1.1])
ylim([-1.1 1.1])
% xlim([-1 5])
% ylim([-3 3])
%3rd Define the coefficients that will define the pd equation
%Problem coefficients
gamma = 1e10-6; %Viscous damping factor
w = pi*2*2.08e106; %Frequency of the US wave
ca = 1497; %Longitudinal speed of sound in water
k = (1)*(w/ca); %Wavenumber

m = 1; 
c = 1;
a = 0;
f = 0;
d = 0;
specifyCoefficients(model,'m',m,'d',d,'c',c,'a',a,'f',f);
%4rd Apply the boundary conditions
bOuter = applyBoundaryCondition(model,'dirichlet','Edge',(1:4),'u',0); 
%appltBoundaryCondition(model_variable, type of BC, RegionType ('Edge'),
%RegionID(1,2,3, 2:4, etc..), Name-Value)

%5th generate a mesh
generateMesh(model,"GeometricOrder","linear")
% generateMesh(model,'Hmax',0.1)
%Hmax:= target maximum mesh edge length 
%GeometricOrder:= A triangle or tetrahedron representing a linear element has nodes at the corners
figure Name 'fig2-2'
pdeplot(model);
axis equal 
title 'Mallado de estructura'
xlim([-1.1 1.1])
ylim([-1.1 1.1])
% xlim([-1 5])
% ylim([-3 3])

%6th Obtain FEM-matrices
FEMatrices = assembleFEMatrices(model,"nullspace"); %assembles finite
%element matrices and imposes boundary conditions using the method specified by bcmethod.
K = FEMatrices.Kc;
B = FEMatrices.B;
M = FEMatrices.M;

%the nullspace bcmethod eliminates the Dirichlet condition using LA. 
%Internally, the toolbox uses the 'nullspace' approach to impose 
%Dirichlet boundary conditions while computing the solution using 
%solvepde and solve.

%7th solve the eigenvalue pde 
r = [0, 1e2]; %Find the eigen values from this range

%Solve the eigenvalue problem by using the eigs function.
% sigma = 1e9; 
% numberEigenvalues = 5;
% [eigenvectorsEigs,eigenvaluesEigs] = eigs(K,M,numberEigenvalues,sigma);
% eigenvaluesEigs = diag(eigenvaluesEigs); %Reshape the diagonal eigenvaluesEigs matrix into a vector.
% % 
% [maxEigenvaluesEigs,maxIndex] = max(eigenvaluesEigs); %Find the largest eigenvalue and its index in the eigenvalues vector.
% eigenvectorsEigs = B*eigenvectorsEigs; %Add the constraint values to get the full eigenvector.
% % 
% %Now, solve the same eigenvalue problem using solvepdeeig. Set the range for solvepdeeig to be slightly larger than the range from eigs.
% r = [min(eigenvaluesEigs)*0.99 max(eigenvaluesEigs)*1.01];
result = solvepdeeig(model,r);
l = length(result.Eigenvalues);
disp(l)

eigenvectorsPde = result.Eigenvectors;
eigenvaluesPde = result.Eigenvalues;


% 
% %Compare the solutions.
% eigenValueDiff = sort(eigenvaluesPde) - sort(eigenvaluesEigs);
% fprintf(['Max difference in eigenvalues' ...
%          ' from solvepdeeig and eigs: %e\n'], ...
%   norm(eigenValueDiff,inf));


%plot
% h.Position = [1 1 2 1].*h.Position;
% subplot(1,2,1)
% axis equal
% % pdeplot(model,'XYData',eigenvectorsEigs(:,maxIndex),'Contour','on')
% title(sprintf('eigs eigenvector, eigenvalue: %12.4e', ...
%                eigenvaluesEigs(maxIndex)))
figure Name 'Results-2'
for j = 1:l
    subplot(2,5,abs(j))
    xlabel('x')
    ylabel('y')
    pdeplot(model,'XYData',eigenvectorsPde(:,j),'Contour','on','ColorMap','parula')
    title(sprintf('EgM %d, Fq: %12.3e Hz',j, ...
               sqrt( eigenvaluesPde(j)^2 *ca^2)))
    xlabel('x')
    ylabel('y')
    xlim([-1.1 1.1])
    ylim([-1.1 1.1])
    % xlim([-1 5])
    % ylim([-3 3])
end

% filename1=(['EigenM_freq_' num2str(w/2*pi) 'kHz' 'Geom' 'Range' num2str(r) ]);
% save(filename1,'X','Y','Z','Pres','lamb','frec','x','y','F0','Phg','xoff','yoff','resX')
%writematrix(model.Mesh.Nodes,'modelmesh.csv','Delimiter',',')
%writematrix(eigenvectorsPde,'eigenvectors.csv','Delimiter',',')
[px,py] = gradient(eigenvectorsPde);

figure Name 'Gradient'
contour(x,y,z)
hold on
quiver(x,y,px,py)
hold off