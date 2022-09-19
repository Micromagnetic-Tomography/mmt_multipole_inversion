import sympy as smp

x, y, z = smp.symbols('x y z', real=True)
r = smp.sqrt(x ** 2 + y ** 2 + z ** 2)
r2 = x ** 2 + y ** 2 + z ** 2

# To simplify
R = smp.symbols('R', real=True)

# -----------------------------------------------------------------------------
# Quadrupole field integrals

# flux_pz12 = smp.integrate(z * (r2 - 5 * (z ** 2)) / r ** 7, x)
# flux_pz12 = smp.integrate(flux_pz12, y)
# flux_pz12 = smp.diff(flux_pz12, z)
# print(flux_pz12)
# print('-' * 80)

flux_pz42 = smp.integrate(y * (r2 - 5 * (z ** 2)) / r ** 7, y)
flux_pz42 = smp.integrate(flux_pz42, x)
# flux_pz42 = smp.integrate((y * z) / r ** 5, x)
# flux_pz42 = smp.integrate(flux_pz42, y)
# flux_pz42 = smp.diff(flux_pz42, z)
smp.pprint(flux_pz42.subs(smp.sqrt(x ** 2 + y ** 2 + z ** 2), R))
print('-' * 80)

flux_pz52 = smp.integrate((x * y) / r ** 5, x)
flux_pz52 = smp.integrate(flux_pz52, y)
flux_pz52 = smp.diff(flux_pz52, z)
smp.pprint(flux_pz52.subs(smp.sqrt(x ** 2 + y ** 2 + z ** 2), R))
