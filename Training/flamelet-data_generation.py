import os
import cantera as ct
import numpy as np
import pandas as pd

def Makedirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def compute_Z_stoich(gas, fuel, oxid, Tin, p):
    s_H, s_O = 1.0, -1.0

    # Oxidizer side
    gas.set_equivalence_ratio(phi=0, fuel=fuel, oxidizer=oxid)
    gas.TP = Tin, p
    Y_H_ox = gas.elemental_mass_fraction('H')
    Y_O_ox = gas.elemental_mass_fraction('O')
    beta_ox = s_H * Y_H_ox + s_O * Y_O_ox

    # Fuel side
    gas.set_equivalence_ratio(phi=1e10, fuel=fuel, oxidizer=oxid)
    gas.TP    = Tin, p
    Y_H_fuel  = gas.elemental_mass_fraction('H')
    Y_O_fuel  = gas.elemental_mass_fraction('O')
    beta_fuel = s_H * Y_H_fuel + s_O * Y_O_fuel

    # （phi = 1）
    gas.set_equivalence_ratio(phi=1.0, fuel=fuel, oxidizer=oxid)
    gas.TP      = Tin, p
    Y_H_stoich  = gas.elemental_mass_fraction('H')
    Y_O_stoich  = gas.elemental_mass_fraction('O')
    beta_stoich = s_H * Y_H_stoich + s_O * Y_O_stoich

    # Z_stoich
    Z_stoich = (beta_stoich - beta_ox) / (beta_fuel - beta_ox)

    return Z_stoich

def get_mixture_fraction_from_equivalence_ratio(equivalence_ratio, Z_stoich):

    """
    This function computes mixture fraction vector based on the equivalence
    ratio and the stoichiometric mixture fraction using:
    equivalence_ratio = Z/(1 - Z) * (1 - Z_stoich)/Z_stoich
    
    Input:
    ----------
    `equivalence_ratio`
               - scalar or vector of equivalence ratio(s).
    `Z_stoich` - stoichiometric mixture fraction.

    Output:
    ----------
    `Z`        - vector of mixture fractions. Each element corresponds to the
                 element in `equivalence_ratio` vector.
    """
    if np.isscalar(equivalence_ratio):
        A = equivalence_ratio * Z_stoich / (1 - Z_stoich)
        Z = A / (1+A)
    elif len(equivalence_ratio) == 1:
        equivalence_ratio = np.asscalar(np.array(equivalence_ratio))
        A = equivalence_ratio * Z_stoich / (1 - Z_stoich)
        Z = A / (1+A)
    else:
        Z = np.empty([len(equivalence_ratio), 1])
        for i in range(0, len(equivalence_ratio)):
            A = equivalence_ratio[i] * Z_stoich / (1 - Z_stoich)
            Z[i] = A / (1+A)

    return Z

fuel = 'H2'
oxid = 'O2:0.21, N2:0.79'

pres = 1.0
temp = 300
equi_list = [i/100 for i in range(35, 70+1, 1)]

gas = ct.Solution('Glarborg/glarborg_21sp.yaml')

Makedirs('results')

for equi in equi_list:
    p   = ct.one_atm * pres
    Tin = temp

    gas.set_equivalence_ratio(phi=equi, fuel=fuel, oxidizer=oxid)
    gas.TP = Tin, p

    width    = 0.20 #m
    loglevel = 0

    f = ct.FreeFlame(gas, width=width)
    f.transport_model = 'multicomponent'
    f.soret_enabled = True

    f.max_time_step_count = 3000
    f.set_refine_criteria(ratio=3, slope=0.03, curve=0.03)
    f.solve(loglevel)

    lbv = float(f.velocity[0]*100)
    print(f'phi {equi :.2f}: LBV(multi) = {lbv:.2f} cm/s')

    x = f.grid
    species = gas.species_names
    n_species = len(species)
    n_points = len(x)

    data = {'x (m)': x, 'T (K)': f.T, 'u (m/s)': f.velocity, 'density (kg/m3)': f.density}

    for i, sp in enumerate(species):
        data[f'Y_{sp}'] = f.Y[i, :]

    net_rates_mol = np.zeros((n_species, n_points))  # kmol/m3/s
    net_rates_mass = np.zeros((n_species, n_points))  # kg/m3/s
    molecular_weights = np.array(gas.molecular_weights)

    for j in range(n_points):
        gas.TDY = f.T[j], f.density[j], f.Y[:, j]
        net_rates_mol[:, j]  = gas.net_production_rates
        net_rates_mass[:, j] = gas.net_production_rates * molecular_weights

    for i, sp in enumerate(species):
        data[f'Wdot_mass_{sp} [kg/m3/s]'] = net_rates_mass[i, :]


    n_points = len(f.grid)
    phi_local = np.zeros(n_points)

    # stoichiometry
    gas.set_equivalence_ratio(phi=1.0, fuel=fuel, oxidizer=oxid)
    H_stoich = gas.elemental_mass_fraction('H')
    O_stoich = gas.elemental_mass_fraction('O')
    stoich_ratio = H_stoich / O_stoich

    for j in range(n_points):
        gas.TDY = f.T[j], f.density[j], f.Y[:, j]
        H = gas.elemental_mass_fraction('H')
        O = gas.elemental_mass_fraction('O')
        if O > 0:
            phi_local[j] = (H/O) / stoich_ratio
        else:
            phi_local[j] = np.nan
    #print(phi_local)

    Z_stoich = compute_Z_stoich(gas, fuel, oxid, Tin, p)
    Z        = get_mixture_fraction_from_equivalence_ratio(phi_local, Z_stoich)

    #print(Z)

    # 🔽 ここを追加！
    data['Local Equivalence ratio'] = phi_local
    data['Mixture fraction (Z)'] = Z.flatten()

    df = pd.DataFrame(data)
    df.to_csv(f'results/p{pres}-T{temp}-phi{equi:.2f}.csv', index=False)
