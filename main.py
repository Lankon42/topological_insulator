import kwant
import kwant.continuum
import numpy as np
import scipy
import scipy.sparse.linalg as sla
from tqdm import tqdm
from matplotlib import pyplot
from scipy import constants

a = 0.1  # шаг решетки
r1 = 3  # радиус первой окружности
r2 = 5  # радиус второй окружности
r3 = 8  # 2*r3 длина квадрата

m_e = scipy.constants.m_e / (scipy.constants.eV * 1e-3)  # масса электрона
hbar = scipy.constants.hbar / \
    (scipy.constants.eV * 1e-3)  # постоянная Планка / 2pi
q_e = scipy.constants.e  # заряд

U1 = 20  # добавка к гамильтониану треугольника
U2 = 5  # добавка к гамильтониану первой окружности
epsilon = 0.1  # хотим, чтобы добавка на границе второй окружности была <= данной константе
# коэффициент экспоненты в добавке к гамильтониану к кольцу
alpha = np.log(U2 / epsilon) / (r2 - r1)


def exp_(x, y, off_x=0, off_y=0):
    r = np.sqrt((x - off_x)**2 + (y - off_y)**2)
    return U2 / np.exp(alpha * (r - r1))


def get_shapes(off_x=0, off_y=0):
    sqr3 = np.sqrt(3, dtype=np.float64)  # корень из 3

    # Прямоугольник
    def rectangular(site):
        (x, y) = site.pos
        rsq = (x - off_x) ** 2 + (y - off_y) ** 2
        return rsq >= r2 ** 2 and -r3 <= y <= r3 and -r3 <= x <= r3

    rectangular_in_site = (r3, r3)

    # Кольцо
    def ring(site):
        (x, y) = site.pos
        x -= off_x
        y -= off_y
        rsq = x ** 2 + y ** 2
        return r1 ** 2 < rsq < r2 ** 2

    ring_in_site = (off_x, off_y + r2 - 1)

    # Круг
    def circle(site):
        (x, y) = site.pos
        x -= off_x
        y -= off_y
        rsq = x ** 2 + y ** 2
        # внешняя часть треугольника
        b = not (y <= (sqr3 * x + r1) and y <=
                 (-sqr3 * x + r1) and (y >= -r1 / 2))
        return b and rsq <= r1 ** 2

    circle_in_site1 = (off_x, off_y - r1 + 1)
    circle_in_site2 = (off_x - r1 + 1, off_y)
    circle_in_site3 = (off_x + r1 - 1, off_y)

    # Треугольник
    def triangle(site):
        (x, y) = site.pos
        x -= off_x
        y -= off_y
        # a = r1 * sqr3 длина стороны треугольника
        # внутренность треугольника
        b = (y <= (sqr3 * x + r1) and y <= (-sqr3 * x + r1) and (y >= -r1 / 2))
        return b

    triangle_in_site = (off_x, off_y)

    return {
        'rectangular': (rectangular, rectangular_in_site),
        'circle1': (circle, circle_in_site1),
        'circle2': (circle, circle_in_site2),
        'circle3': (circle, circle_in_site3),
        'triangle': (triangle, triangle_in_site),
        'ring': (ring, ring_in_site)
    }


def get_templates(off_x=10, off_y=10):
    # Hamiltonian with constants
    ham_rectangular = """ hbar**2 / (2 * m_e / 1e18) * (k_x**2 + (k_y + 1e-15 * B * x / hbar) ** 2 )*sigma_0 + E_d*sigma_0 +
                        hbar*alpha*1e9*(k_x*sigma_y - (k_y + 1e-15*B*x/hbar)*sigma_x) + 
                        0*lam*k_x*(k_x**2-3*(k_y + 1e-15*B*x/hbar)**2)*sigma_z {} """
    ham_triangle = ham_rectangular.format("+ U1*sigma_0")
    ham_circle = ham_rectangular.format("+ U2*sigma_0")
    ham_ring = ham_rectangular.format("+ +exp_(x, y, off_x, off_y)*sigma_0")
    ham_rectangular = ham_rectangular.format('+0*sigma_0')
    template_strings = {
        'ham_rec': ham_rectangular,
        'ham_triangle': ham_triangle,
        'ham_circle': ham_circle,
        'ham_ring': ham_ring
    }
    return template_strings


def make_system(center):
    template_strings = get_templates(center[0], center[1])
    template = {k: kwant.continuum.discretize(v, coords=('x', 'y'), grid=a) for k, v in template_strings.items()}

    shapes = get_shapes(center[0], center[1])

    syst = kwant.Builder()

    # Заполняем наши формы нашими гамильтонианами.
    syst.fill(template['ham_rec'], *shapes['rectangular'])
    syst.fill(template['ham_triangle'], *shapes['triangle'])
    syst.fill(template['ham_circle'], *shapes['circle1'])
    syst.fill(template['ham_circle'], *shapes['circle2'])
    syst.fill(template['ham_circle'], *shapes['circle3'])
    syst.fill(template['ham_ring'], *shapes['ring'])

    # Здесь задана трансляционная симметрия вдоль положительного направления x, т. е. провод
    lead1 = kwant.Builder(kwant.TranslationalSymmetry((a, 0)))
    lead1.fill(template['ham_rec'], lambda s: -r3 <= s.pos[1] <= r3, (0, 0))
    syst.attach_lead(lead1)
    syst.attach_lead(lead1.reversed())

    lead2 = kwant.Builder(kwant.TranslationalSymmetry((0, a)))
    lead2.fill(template['ham_rec'], lambda s: -r3 <= s.pos[0] <= r3, (0, 0))
    syst.attach_lead(lead2)
    syst.attach_lead(lead2.reversed())

    syst = syst.finalized()
    return syst


def plot_spectrum(syst, params, Bfields):
    energies = []
    for B in tqdm(Bfields):
        # Obtain the Hamiltonian as a sparse matrix
        params['B'] = B
        ham_mat = syst.hamiltonian_submatrix(params=params, sparse=True)
        # we only calculate the 15 lowest eigenvalues
        ev = sla.eigsh(ham_mat.tocsc(), k=15, sigma=0,
                       return_eigenvectors=False)
        energies.append(ev)

    pyplot.figure()
    pyplot.plot(Bfields, energies)
    pyplot.xlabel("magnetic field [arbitrary units]")
    pyplot.ylabel("energy [t]")
    pyplot.show()
    return energies


def sorted_eigs(ev):
    evals, evecs = ev
    evals, evecs = map(np.array, zip(*sorted(zip(evals, evecs.transpose()))))
    return evals, evecs.transpose()


def plot_wave_function(syst, B=0.5):
    # Calculate the wave functions in the system.
    ham_mat = syst.hamiltonian_submatrix(sparse=True, params=dict(
        B=B, hbar=hbar, m_e=m_e, U1=U1, U2=U2, exp_=exp_))
    evals, evecs = sorted_eigs(sla.eigsh(ham_mat.tocsc(), k=20, sigma=0))

    # Plot the probability density of the 5th eigenmode.
    kwant.plotter.map(syst, np.abs(
        evecs[:, 5])**2, colorbar=False, oversampling=1)


def ldosE(syst, params, energy=100):
    ldos = kwant.ldos(syst, energy=energy, params=params)
    kwant.plotter.map(syst, ldos[0::2]+ldos[1::2], cmap='viridis')


def ldosX(syst, params, energies):
    ldos = np.empty((len(energies), 2 * (2 * int(r3 / a) + 1) ** 2))
    for i in tqdm(range(len(energies))):
        ldos[i] = kwant.ldos(syst, energy=energies[i], params=params)
    pyplot.plot(energies, ldos[:, 0]+ldos[:, 1])  # в самом первом узле
    pyplot.show()


def DOS(syst, params, Bfields, energies, filen="test"):
    lldos = np.empty((len(energies), len(Bfields), 2 * (2 * int(r3 / a) + 1) ** 2))
    for i in tqdm(range(len(energies))):
        for j in range(len(Bfields)):
            params['B'] = Bfields[j]
            lldos[i, j] = kwant.ldos(syst, energy=energies[i], params=params)

    with open(filen, 'wb') as f:
        np.save(f, lldos)


def E_B(n=1, B=0, E_d=0, mu=1):
    return E_d + np.sqrt(2*hbar*2*mu/m_e*n*B)


def draw(filen, magn, enrg):
    with open(filen, 'rb') as f:
        lldos = np.load(f)
    lldos1 = np.zeros((len(enrg), len(magn)))
    for i in range(len(enrg)):
        for j in range(len(magn)):
            lldos1[i, j] = np.mean(lldos[i, j])

    pyplot.figure(figsize=(10, 100))
    pyplot.pcolormesh(magn, enrg, lldos1, cmap='jet', shading='auto')
    pyplot.xlabel('B')
    pyplot.ylabel('E')
    pyplot.colorbar()
    pyplot.show()


def main():
    # Поля от 0 до 7, 10
    _params = dict(B=10, alpha=471200, E_d=-10, hbar=hbar, m_e=m_e, U1=U1,
                   U2=U2, exp_=exp_, off_x=0, off_y=0)
    centers = np.array([[0, 0], [30, 30]])

    syst = make_system(centers[0])
    # kwant.plot(syst)

    # Check that the system looks as intended. Colour according to the on-site Hamiltonian of site i
    # kwant.plotter.map(syst, lambda i: syst.hamiltonian(i, i, params=_params), cmap='jet')

    # Eigen values
    # plot_spectrum(syst, _params, [iB * 0.5 for iB in range(100)])

    # Plot the probability density of the 10th eigenmode
    # plot_wave_function(syst)

    # ldosE(syst, _params, 30)
    ldosX(syst, _params, [i * 0.5 for i in range(400)])
    magn = [30 + iB * 0.5 for iB in range(10)]
    enrg = [iE * 0.5 for iE in range(100)]
    # DOS(syst, _params, magn, enrg, "test3003_5")
    # draw("test3003_5", magn, enrg)


if __name__ == '__main__':
    main()
