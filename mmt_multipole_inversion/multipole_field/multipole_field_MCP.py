# Multipole field expressions using Maxwell Cartesian Polynomials
#
# 
import numpy as np


def Br_field_octupole(r, theta, phi, W1, W2, W3, W4, W5, W6, W7):
    r"""
    Compute the octupole field at a given position (r, theta, phi) in
    spherical coordinates, and using 7 octupole moments::

        W1 -> W_xxx
        W2 -> W_xxy
        W3 -> W_xxz
        W4 -> W_xyz
        W5 -> W_yyx
        W6 -> W_yyy
        W7 -> W_yyz

    We are assuming Ms lies in these octupole moments, so Bx is::

        Bx =  mu0    _5_
              ----   \    W_i * Px_i
              4 PI   /__
                     i=1

    where Px_i is the polynomial associated to the oct moment W_i, for the
    x-component of the B field. Similarly for the other components.

    """
    # TODO: Speed up the calculations vectorizing the operations!

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x2, y2, z2 = x ** 2, y ** 2, z ** 2
    r2 = r ** 2
    r9 = r2 * r2 * r2 * r2 * r

    f = 1e-7  # mu0 / 4 PI

    Bx =  W1 * 5 * (7 * x2 * (x2 - 3 * z2) + 3 * r2 * (z2 - x2))
    Bx += W2 * 15 * x * y * (7 * (x2 - z2) - 2 * r2)
    Bx += W3 * 5 * x * z * (7 * (3 * x2 - z2) - 6 * r2)
    Bx += W4 * 30 * y * z * (7 * x2 - r2)
    Bx += W5 * 15 * (y2 - z2) * (7 * x2 - r2)
    Bx += W6 * 35 * x * y * (y2 - 3 * z2)
    Bx += W7 * 35 * x * z * (3 * y2 - z2)

    #
    By  = W1 * 35 * x * y * (x2 - 3 * z2)
    By += W2 * 15 * (x2 - z2) * (7 * y2 - r2)
    By += W3 * 35 * y * z * (3 * x2 - z2)
    By += W4 * 30 * x * z * (7 * y2 - r2)
    By += W5 * 15 * x * y * (7 * (y2 - z2) - 2 * r2)
    By += W6 * 5 * (7 * y2 * (y2 - 3 * z2) + 3 * r2 * (z2 - y2))
    By += W7 * 5 * y * z * (7 * (3 * y2 - z2) - 6 * r2)

    Bz  = W1 * 5 * x * z * (7 * (x2 - 3 * z2) + 6 * r2)
    Bz += W2 * 15 * y * z * (7 * (x2 - z2) + 2 * r2)
    Bz += W3 * 5 * (7 * z2 * (3 * x2 - z2) - 3 * r2 * (x2 - z2))
    Bz += W4 * 30 * x * y * (7 * z2 - r2)
    Bz += W5 * 15 * x * z * (7 * (y2 - z2) + 2 * r2)
    Bz += W6 * 5 * y * z * (7 * (y2 - 3 * z2) + 6 * r2)
    Bz += W7 * 5 * (7 * z2 * (3 * y2 - z2) - 3 * r2 * (y2 - z2))

    # Get the radial field component from the Cartesian (Bx,By,Bz) vector
    Br = f * (1 / r9) * (Bx * np.sin(theta) * np.cos(phi) +
                         By * np.sin(theta) * np.sin(phi) +
                         Bz * np.cos(theta))

    return Br


def Br_field_quadrupole_Cartesian(x, y, z, Q1, Q2, Q3, Q4, Q5):
    r"""
    Compute the quadrupole field at a given position (r, theta, phi) in
    spherical coordinates, and using 5 quadrupole moments::

        Q1 -> Theta_xx
        Q2 -> Theta_xy
        Q3 -> Theta_xz
        Q4 -> Theta_yy
        Q5 -> Theta_yz

    We are assuming Ms lies in these quadrupole moments, so Bx is::

        Bx =  mu0    _5_
              ----   \    Q_i * Px_i
              4 PI   /__
                     i=1

    where Px_i is the polynomial associated to the quad moment Q_i, for the
    x-component of the B field. Similarly for the other components.

    """
    # TODO: Speed up the calculations vectorizing the operations!

    # Not necessary: dip_r, pos_r
    # Args: r, theta, phi

    # r = pos_r - dip_r
    # x, y, z = dr[0], dr[1], dr[2]
    # rho2 = np.sum(dr ** 2, axis=1)
    # rho = np.sqrt(rho2)

    #     x = r * np.sin(theta) * np.cos(phi)
    #     y = r * np.sin(theta) * np.sin(phi)
    #     z = r * np.cos(theta)
    x2, y2, z2 = x ** 2, y ** 2, z ** 2
    r2 = x2 + y2 + z2
    r = np.sqrt(r2)
    r7 = r2 * r2 * r2 * r

    f = 1e-7  # mu0 / 4 PI

    Bx = Q1 * x * (5 * (x2 - z2) - 2 * r2)
    Bx += Q2 * y * 2 * (5 * x2 - r2)
    Bx += Q3 * z * 2 * (5 * x2 - r2)
    Bx += Q4 * 5 * x * (y2 - z2)
    Bx += Q5 * 10 * x * y * z

    #
    By = Q1 * 5 * y * (x2 - z2)
    By += Q2 * x * 2 * (5 * y2 - r2)
    By += Q3 * 10 * x * y * z
    By += Q4 * y * (5 * (y2 - z2) - 2 * r2)
    By += Q5 * z * 2 * (5 * y2 - r2)

    Bz = Q1 * (5 * z * (x2 - z2) + 2 * r2 * z)
    Bz += Q2 * 10 * x * y * z
    Bz += Q3 * 2 * x * (5 * z2 - r2)
    Bz += Q4 * (5 * z * (y2 - z2) + 2 * r2 * z)
    Bz += Q5 * 2 * y * (5 * z2 - r2)

    return f * (1 / r7) * np.array([Bx, By, Bz])


def Br_field_quadrupole(r, theta, phi, Q1, Q2, Q3, Q4, Q5):
    r"""
    Compute the quadrupole field at a given position (r, theta, phi) in
    spherical coordinates, and using 5 quadrupole moments::

        Q1 -> Theta_xx
        Q2 -> Theta_xy
        Q3 -> Theta_xz
        Q4 -> Theta_yy
        Q5 -> Theta_yz

    We are assuming Ms lies in these quadrupole moments, so Bx is::

        Bx =  mu0    _5_
              ----   \    Q_i * Px_i
              4 PI   /__
                     i=1

    where Px_i is the polynomial associated to the quad moment Q_i, for the
    x-component of the B field. Similarly for the other components.

    """
    # TODO: Speed up the calculations vectorizing the operations!
    # TODO: Use the Cartesian function

    # Not necessary: dip_r, pos_r
    # Args: r, theta, phi

    # r = pos_r - dip_r
    # x, y, z = dr[0], dr[1], dr[2]
    # rho2 = np.sum(dr ** 2, axis=1)
    # rho = np.sqrt(rho2)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x2, y2, z2 = x ** 2, y ** 2, z ** 2
    r2 = r ** 2
    r7 = r2 * r2 * r2 * r

    f = 1e-7  # mu0 / 4 PI

    Bx = Q1 * x * (5 * (x2 - z2) - 2 * r2)
    Bx += Q2 * y * 2 * (5 * x2 - r2)
    Bx += Q3 * z * 2 * (5 * x2 - r2)
    Bx += Q4 * 5 * x * (y2 - z2)
    Bx += Q5 * 10 * x * y * z

    #
    By = Q1 * 5 * y * (x2 - z2)
    By += Q2 * x * 2 * (5 * y2 - r2)
    By += Q3 * 10 * x * y * z
    By += Q4 * y * (5 * (y2 - z2) - 2 * r2)
    By += Q5 * z * 2 * (5 * y2 - r2)

    Bz = Q1 * (5 * z * (x2 - z2) + 2 * r2 * z)
    Bz += Q2 * 10 * x * y * z
    Bz += Q3 * 2 * x * (5 * z2 - r2)
    Bz += Q4 * (5 * z * (y2 - z2) + 2 * r2 * z)
    Bz += Q5 * 2 * y * (5 * z2 - r2)

    # Get the radial field component from the Cartesian (Bx,By,Bz) vector
    Br = f * (1 / r7) * (Bx * np.sin(theta) * np.cos(phi) +
                         By * np.sin(theta) * np.sin(phi) +
                         Bz * np.cos(theta))

    return Br


def Br_field_dipole(r, theta, phi, m1, m2, m3):
    r"""
    Compute the dipole field at a given position (r, theta, phi) in
    spherical coordinates, and using 3 dipolar moments.

    We are assuming Ms lies in these dipole moments, so Bx is::

        Bx =  mu0    _3_
              ----   \    m_i * Px_i
              4 PI   /__
                     i=1

    where Px_i is the polynomial associated to the dipole moment m_i, for the
    x-component of the B field. Similarly for the other components.

    """
    # TODO: Speed up the calculations vectorizing the operations!

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x2, y2, z2 = x ** 2, y ** 2, z ** 2
    r2 = r ** 2
    r5 = r2 * r2 * r

    f = 1e-7  # mu0 / 4 PI

    Bx = m1 * (3 * x2 - r2)
    Bx += m2 * (3 * y * x)
    Bx += m3 * (3 * z * x)

    By = m1 * (3 * y * x)
    By += m2 * (3 * y2 - r2)
    By += m3 * (3 * y * z)

    Bz = m1 * (3 * z * x)
    Bz += m2 * (3 * z * y)
    Bz += m3 * (3 * z2 - r2)

    # Get the radial field component from the Cartesian (Bx,By,Bz) vector
    Br = f * (1 / r5) * (Bx * np.sin(theta) * np.cos(phi) +
                         By * np.sin(theta) * np.sin(phi) +
                         Bz * np.cos(theta))

    return Br
