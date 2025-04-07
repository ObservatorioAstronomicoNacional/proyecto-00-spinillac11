import numpy as np
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt


# Parámetros orbitales
GM = 398600.4405  
R_EARTH = 6371.0

a = 1.30262 * R_EARTH
e = 0.16561
omega_deg = 15.0
omega = np.radians(omega_deg) # pasar a radianes

tp = Time("2025-03-31T00:00:00", scale='utc')

epsilon_E = 1e-8 #Tolerancia newton raphson
delta_r = 1.28342948e-6 #Incertidumbre posición

#######################################################
# Funciones

# Ecuación de Kepler
def kepler_equation(E, l, e):
    return E - e * np.sin(E) - l

def kepler_deriv(E, e):
    return 1 - e * np.cos(E)

# Implementación newton-raphson
def solve_kepler(e, l, eps=epsilon_E, max_iter=500):
    E = l 
    for i in range(max_iter):
        f = kepler_equation(E, l, e)
        fp = kepler_deriv(E, e)
        if abs(fp) < 1e-12:
            raise RuntimeError("Derivada cercana a cero")
        E_new = E - f/fp
        if abs(E_new - E) < eps:
            return E_new
        E = E_new
    raise RuntimeError(f"No convergió en {max_iter} iteraciones")

# Cálculo de posición
def position(t_obs):
    delta_t = (t_obs - tp).to_value('s')
    n = np.sqrt(GM / a**3)
    l = (n * delta_t) % (2 * np.pi) # Normalizar al intervalo 0 - 2pi
    
    E = solve_kepler(e, l)
    
    f = 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E/2))
    
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    phi = f + omega
    return r, phi

# Gráfico de órbita
def orbit():
    num_points=1000
    T = 2 * np.pi * np.sqrt(a**3 / GM) #periodo
    segundos = np.linspace(0, T, num_points + 1)  
    times = tp + TimeDelta(segundos, format='sec')
    
    coords = [position(ti) for ti in times]  # Lista de (r, phi)
    
    r = np.array([r for r, phi in coords])
    phi = np.array([phi for r, phi in coords])
    x = r*np.cos(phi) 
    y = r*np.sin(phi) 
    
    plt.figure()
    plt.plot(x, y, '-', lw=1)
    plt.scatter(0, 0, c='k', s=20, label='Tierra')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.title('Órbita del satélite')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

# Búsqueda de fecha
def date(r0, inc=delta_r):
    num_points = 1000 
    T = 2 * np.pi * np.sqrt(a**3 / GM)  # Periodo orbital
    
    segundos = np.linspace(0, T, num_points + 1)
    times = tp + TimeDelta(segundos, format='sec') 
    
    # Obtener radios para todos los tiempos 
    radios = np.array([position(ti)[0] for ti in times])  
    diffs = radios - r0  
    
     
    # Buscar cambio de signos
    sign_changes = []
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i+1] < 0:
            sign_changes.append(i)  # Almacenar índices de cruces
    
    # Si no hay cruces, lanzar error
    if len(sign_changes) == 0:
        raise ValueError(f"r0 = {r0} km no está en la órbita")
    
    # Tomar el primer cruce
    i = sign_changes[0]
    t1, t2 = times[i], times[i+1]
    
    # Bisección para precisar el tiempo t0
    for i in range(50):
        tm = t1 + (t2 - t1) * 0.5
        r_tm = position(tm)[0]
        
        if abs(r_tm - r0) < inc:
            return tm

        if (r_tm - r0) > 0:
            t1 = tm
        else:
            t2 = tm
    
    return tm  # Aproximación final
        
######################################################
# Pruebas
# Prueba de posición
t_test = Time("2025-04-01T00:00:00", scale='utc')
r_test, phi_test = position(t_test)

print(f"Para la fecha {t_test.iso} la posición es:")
print(f"r(t) = {r_test:.6f} km")
print(f"phi(t) = {np.degrees(phi_test):.6f}°\n")
    
# Prueba de fecha
r0 = 1.5 * R_EARTH
try:
    t0 = date(r0)

    print(f"La posición radial r = 1.5R es alcanzada en la fecha:")
    print(f"t0 = {t0.iso}")
except ValueError as err:
        print(f"Error: {str(err)}")
    
# Gráfico
orbit()
