# Given optical integrands extract i-j-k force components
from optic_integrands import F_Integrand, F_Integrand_GRAD, F_Integrand_SCATT

def F_INTEGRAND_X(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFx = F_Integrand(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[0]
    return dFx
def F_INTEGRAND_Y(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFy = F_Integrand(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[1]
    return dFy
def F_INTEGRAND_Z(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFz = F_Integrand(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[2]
    return dFz
def F_INTEGRAND_GRAD_X(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFx = F_Integrand_GRAD(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[0]
    return dFx
def F_INTEGRAND_GRAD_Y(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFy = F_Integrand_GRAD(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[1]
    return dFy
def F_INTEGRAND_GRAD_Z(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFz = F_Integrand_GRAD(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[2]
    return dFz
def F_INTEGRAND_SCATT_X(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFx = F_Integrand_SCATT(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[0]
    return dFx
def F_INTEGRAND_SCATT_Y(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFy = F_Integrand_SCATT(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[1]
    return dFy
def F_INTEGRAND_SCATT_Z(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct):
    dFz = F_Integrand_SCATT(theta,phi,x,y,a,f,w0,PI,theta0,n_t,n_i,struct)[2]
    return dFz