# Multipole field expressions using the Spherical Harmonics Basis
#
#
import numpy as np


def Br_field_octupole(r, theta, phi, W1, W2, W3, W4, W5, W6, W7):
    """
    Compute the octupole field at a given position (r, theta, phi) in
    spherical coordinates, and using 7 octupole moments::

        W1 ->
        W2 ->
        W3 ->
        W4 ->
        W5 ->
        W6 ->
        W7 ->

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

    Bx = W1 * np.sqrt(2.5) * x * z * (-3 * r2 + 7 * z2)
    Bx += W2 * -(np.sqrt(0.6) * (4 * (r2 ** 2) + 35 * z2 * (y2 + z2)
                                 - 5 * r2 * (y2 + 7 * z2))) / 2.
    Bx += W3 * -(np.sqrt(15) * x * y * (r2 - 7 * z2)) / 2.
    Bx += W4 * np.sqrt(1.5) * x * z*(5 * r2 - 7*(2 * y2 + z2))
    Bx += W5 * -(np.sqrt(6) * y * z * (-6 * x2 + y2 + z2))
    Bx += W6 * (4 * x**4 + 3 * y2 * (y2 + z2) - 3 * x2 * (7 * y2 + z2)) / 2.
    Bx += W7 * -(x * y * (-15 * r2 + 28 * y2 + 21 * z2)) / 2.

    #
    By = W1 * np.sqrt(2.5) * y * z * (-3 * r2 + 7 * z2)
    By += W2 * -(np.sqrt(15) * x * y * (r2 - 7 * z2))/2.
    By += W3 * (np.sqrt(0.6) * ((r2 ** 2) + 35 * y2 * z2
                                - 5 * r2 * (y2 + z2))) / 2.
    By += W4 * np.sqrt(1.5) * y * z * (9 * r2 - 7 * (2 * y2 + z2))
    By += W5 * -(np.sqrt(6) * x * (r2 - 7 * y2) * z)
    By += W6 * (x * y * (13 * r2 - 7 * (4 * y2 + z2)))/2.
    By += W7 * (-3 * (r2 ** 2) + 3 * r2 * (9 * y2 + z2)
                - 7 * (4 * y**4 + 3 * y2 * z2)) / 2.

    Bz = W1 * (3 * (r2 ** 2) + 35 * (z2 ** 2) - 30 * r2 * z2) / np.sqrt(10)
    Bz += W2 * (np.sqrt(15) * x * z * (-3 * r2 + 7 * z2)) / 2.
    Bz += W3 * (np.sqrt(15) * y * z * (-3 * r2 + 7 * z2)) / 2.
    Bz += W4 * -(np.sqrt(1.5) * (x - y) * (x + y) * (r2 - 7 * z2))
    Bz += W5 * -(np.sqrt(6) * x * y * (r2 - 7 * z2))
    Bz += W6 * (7 * x * (x2 - 3 * y2) * z) / 2.
    Bz += W7 * (-7 * y * (-3 * x2 + y2) * z) / 2.

    # Get the radial field component from the Cartesian (Bx,By,Bz) vector
    Br = f * (1 / r9) * (Bx * np.sin(theta) * np.cos(phi) +
                         By * np.sin(theta) * np.sin(phi) +
                         Bz * np.cos(theta))

    return Br


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

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x2, y2, z2 = x ** 2, y ** 2, z ** 2
    r2 = r ** 2
    r7 = r2 * r2 * r2 * r

    f = 1e-7  # mu0 / 4 PI

    Bx = Q1 * -(np.sqrt(1.5) * x * (r2 - 5 * z2))
    Bx += Q2 * -(np.sqrt(2) * z * (-4 * x2 + y2 + z2))
    Bx += Q3 * 5 * np.sqrt(2) * x * y * z
    Bx += Q4 * (x * (3 * r2 - 5 * (2 * y2 + z2))) / np.sqrt(2)
    Bx += Q5 * -(np.sqrt(2) * y * (-4 * x2 + y2 + z2))

    #
    By = Q1 * -(np.sqrt(1.5) * y * (r2 - 5 * z2))
    By += Q2 * 5 * np.sqrt(2) * x * y * z
    By += Q3 * -(np.sqrt(2) * (r2 - 5 * y2) * z)
    By += Q4 * (y * (7 * r2 - 5 * (2 * y2 + z2))) / np.sqrt(2)
    By += Q5 * -(np.sqrt(2) * x * (r2 - 5 * y2))

    Bz = Q1 * np.sqrt(1.5) * z * (-3 * r2 + 5 * z2)
    Bz += Q2 * -(np.sqrt(2) * x * (r2 - 5 * z2))
    Bz += Q3 * -(np.sqrt(2) * y * (r2 - 5 * z2))
    Bz += Q4 * (5 * (x - y) * (x + y) * z) / np.sqrt(2)
    Bz += Q5 * 5 * np.sqrt(2) * x * y * z

    # Get the radial field component from the Cartesian (Bx,By,Bz) vector
    Br = f * (1 / r7) * (Bx * np.sin(theta) * np.cos(phi) +
                         By * np.sin(theta) * np.sin(phi) +
                         Bz * np.cos(theta))

    return Br


def Br_field_dipole(r, theta, phi, m1, m2, m3):
    """
    Compute the dipole field at a given position (r, theta, phi) in
    spherical coordinates, and using 3 dipolar moments

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


def Br_field_quadrupole_Cartesian(x, y, z, Q1, Q2, Q3, Q4, Q5):
    """
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

    x2, y2, z2 = x ** 2, y ** 2, z ** 2
    r2 = x2 + y2 + z2
    r = np.sqrt(r2)
    r7 = r2 * r2 * r2 * r

    f = 1e-7  # mu0 / 4 PI

    Bx = Q1 * -(np.sqrt(1.5) * x * (r2 - 5 * z2))
    Bx += Q2 * -(np.sqrt(2) * z * (-4 * x2 + y2 + z2))
    Bx += Q3 * 5 * np.sqrt(2) * x * y * z
    Bx += Q4 * (x * (3 * r2 - 5 * (2 * y2 + z2))) / np.sqrt(2)
    Bx += Q5 * -(np.sqrt(2) * y * (-4 * x2 + y2 + z2))

    #
    By = Q1 * -(np.sqrt(1.5) * y * (r2 - 5 * z2))
    By += Q2 * 5 * np.sqrt(2) * x * y * z
    By += Q3 * -(np.sqrt(2) * (r2 - 5 * y2) * z)
    By += Q4 * (y * (7 * r2 - 5 * (2 * y2 + z2))) / np.sqrt(2)
    By += Q5 * -(np.sqrt(2) * x * (r2 - 5 * y2))

    Bz = Q1 * np.sqrt(1.5) * z * (-3 * r2 + 5 * z2)
    Bz += Q2 * -(np.sqrt(2) * x * (r2 - 5 * z2))
    Bz += Q3 * -(np.sqrt(2) * y * (r2 - 5 * z2))
    Bz += Q4 * (5 * (x - y) * (x + y) * z) / np.sqrt(2)
    Bz += Q5 * 5 * np.sqrt(2) * x * y * z

    return f * (1 / r7) * np.array([Bx, By, Bz])
