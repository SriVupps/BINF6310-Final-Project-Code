from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Filename: H_bond_pipeline.py
The aim of this program is to identify hydrogen bonds between a ligand and select residues in a protein, with the 
broader aim of replicating a pipeline for the BINF6310 final project
Author: Sriya Vuppala
"""


def calculate_angle(atom1, atom2, atom3):
    """
    Calculates the angle between three atoms using the dot product
    :param atom1: an input atom dict {"id": string, "type": string: "coord": numpy array}
    :param atom2: an input atom dict {"id": string, "type": string: "coord": numpy array}
    :param atom3: an input atom dict {"id": string, "type": string: "coord": numpy array}
    :return: the angle in degrees (float)
    """
    # initializing vectors
    vector1 = atom1['coord'] - atom2['coord']
    vector2 = atom3['coord'] - atom2['coord']

    # computing dot prod
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    # converting angle in radians to angle in degrees
    angle_radians = np.arccos(dot_product / norm_product)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def euclidean_distance(atom1, atom2):
    """
    Calculates the euclidean distance between the coordinates of 2 atoms
    :param atom1: an input atom dict {"id": string, "type": string: "coord": numpy array}
    :param atom2: an input atom dict {"id": string, "type": string: "coord": numpy array}
    :return: the euclidean distance between the two input atoms (float)
    """
    # accessing atom coordinates
    coord1 = atom1['coord']
    coord2 = atom2['coord']

    # computing euclidean distance between atom coordinates

    distance = np.sqrt(np.sum((coord2 - coord1) ** 2))
    return distance


def find_ligand_residue_range(pdb_file_path, chain_id='N'):
    """
    Finds the range of residues that identify the ligand
    :param pdb_file_path: a string representing the file path to the pdb file
    :param chain_id: a string representing the chain id of the ligand molecule
    :return: a list representing the range of residues of the ligand
    """
    ligand_residues = set()

    # open pdb file
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            # check for lines that start with ATOM
            if line.startswith('ATOM'):
                # check the chain specified in the line
                chain = line[21]
                if chain == chain_id:
                    # extract residue number from the line
                    residue_number = int(line[22:26].strip())
                    ligand_residues.add(residue_number)
        # return range of residues
        return [min(ligand_residues), max(ligand_residues)]


# source: https://biopython-cn.readthedocs.io/zh-cn/latest/en/chr11.html
# source: https://biopython.org/docs/1.75/api/Bio.PDB.NeighborSearch.html
def is_donor(atom, ns):
    """
    Determines if a given atom could be a H bond donor
    :param atom: an input atom dict {"id": string, "type": string: "coord": numpy array}
    :param ns: a NeighborSearch object that will determine atom connectivity
    :return: boolean
    """

    # check if the atom is very electronegative
    return atom['type'] in ['N', 'F', 'O'] and any(
        # check if atom is covalently bonded to H
        neighbor.get_id()[0] == 'H' for neighbor in ns.search(atom['coord'], 1.5, level='A')
    )


def is_acceptor(atom, ns):
    """
    Determines if a given atom could be a H bond acceptor
    :param atom: an input atom dict {"id": string, "type": string: "coord": numpy array}
    :param ns: a NeighborSearch object that will determine atom connectivity
    :return: boolean
    """
    # check if atom is H
    return atom['type'] == 'H' and any(
        # check if atom is covalently bonded to very electronegative atom
        neighbor.get_id()[0] in ['N', 'F', 'O'] for neighbor in ns.search(atom['coord'], 1.5, level='A')
    )


# source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8261469/#
# :~:text=Hydrogen%20bond%20criteria,be%20larger%20than%20100%C2%B0.
def calculate_h_bonds(pdb_file_path, substrate_residue_range, target_residue_range,
                      distance_cutoff=3.5, angle_cutoff=(100, 180)):
    """
    Calculates the lengths of hydrogen bonds between the substrate_residue_range and target_residue_range
    :param pdb_file_path: a string representing the file path to the pdb file
    :param substrate_residue_range: a list with the min and max residues of the substrate
    :param target_residue_range: a list of ints representing target residues
    :param distance_cutoff: float - 3.5 angstroms (the max length of an H bond)
    :param angle_cutoff: tuple representing the range of degrees for h bond angle
    :return: h_bond_dist: list of the lengths (floats) of identified hydrogen bonds
    """
    # parsing the pdb file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)

    # connectivity information
    ns = NeighborSearch(list(structure.get_atoms()))

    # getting the atoms of the substrate/ligand
    substrate_atoms = []
    for s_atom in structure.get_atoms():
        if substrate_residue_range[0] <= s_atom.get_parent().id[1] <= substrate_residue_range[1]:
            substrate_atom_dict = {
                'id': s_atom.get_id(),
                'type': s_atom.element,
                'coord': s_atom.get_coord(),
                'residue': s_atom.get_parent().id[1]
            }
            substrate_atoms.append(substrate_atom_dict)

    # getting the atoms of the selected protein residue range
    target_atoms = []
    for t_atom in structure.get_atoms():
        if target_residue_range[0] <= t_atom.get_parent().id[1] <= target_residue_range[1]:
            target_atom_dict = {
                'id': t_atom.get_id(),
                'type': t_atom.element,
                'coord': t_atom.get_coord(),
                'residue': t_atom.get_parent().id[1]
            }
            target_atoms.append(target_atom_dict)

    # initializing empty lists to store h bonds and h bond distances
    hydrogen_bonds = []
    h_bond_dist = []

    for substrate_atom in substrate_atoms:
        for target_atom in target_atoms:

            # checks if an atom from the ligand is a donor and an atom from the protein is an acceptor
            if is_donor(substrate_atom, ns) and is_acceptor(target_atom, ns):
                # find covalently bonded atoms within a certain distance
                neighbors = ns.search(target_atom['coord'], 1.5, level='A')
                for neighbor in neighbors:
                    if neighbor.get_id()[0] in ["N", "O", "F"]:
                        # calculate distance between atoms
                        distance = euclidean_distance(target_atom, substrate_atom)
                        # calculates angle between 3 atoms
                        angle = calculate_angle(substrate_atom, target_atom, {"coord": neighbor.get_coord()})
                        # determines if it meets criteria to be h bond
                        if distance <= distance_cutoff and angle_cutoff[0] <= angle <= angle_cutoff[1]:
                            hydrogen_bonds.append((substrate_atom, target_atom))
                            h_bond_dist.append(distance)

            # checks if an atom from the ligand is an acceptor and an atom from the protein is a donor
            elif is_acceptor(substrate_atom, ns) and is_donor(target_atom, ns):
                # find covalently bonded atoms within a certain distance
                neighbors = ns.search(substrate_atom['coord'], 1.5, level='A')
                for neighbor in neighbors:
                    if neighbor.get_id()[0] in ["N", "O", "F"]:
                        # calculate distance between atoms
                        distance = euclidean_distance(target_atom, substrate_atom)
                        # calculates angle between 3 atoms
                        angle = calculate_angle(target_atom, substrate_atom, {"coord": neighbor.get_coord()})
                        # determines if it meets criteria to be h bond
                        if distance <= distance_cutoff and angle_cutoff[0] <= angle <= angle_cutoff[1]:
                            hydrogen_bonds.append((substrate_atom, target_atom))
                            h_bond_dist.append(distance)
    return h_bond_dist


def avg_h_bond_dist(h_bond_dists):
    """
    Calculates the average h bond length given a list of h bond lengths
    :param h_bond_dists: a list (floats) of h-bond lengths
    :return:
    """

    # returns no distance if there are no h bonds in list
    if len(h_bond_dists) == 0:
        return 0

    total = 0.0
    count = 0

    for dist in h_bond_dists:
        total += dist
        count += 1

    # calculate average
    average = total / count
    return average


# source: https://stackoverflow.com/questions/61852402/
# how-can-i-plot-a-simple-plot-with-seaborn-from-a-python-dictionary
def h_bond_bar_graph(sample_bonds, residues, non_averaged_distances):
    """
    Generates a bar graph with an overlay of scatter plot for non-averaged distances per sample
    :param sample_bonds: a dictionary mapping sample names to avg h bond length (String -> float)
    :param residues: information about residues for the title
    :param non_averaged_distances: a dictionary mapping sample names to lists of non-averaged distances
    """
    keys = list(sample_bonds.keys())
    vals = [float(sample_bonds[k]) for k in keys]

    fig, ax = plt.subplots()

    # making bar plot for averaged distances
    sns.barplot(x=keys, y=vals, ax=ax, color='blue')
    plt.xlabel("Sample Type")
    plt.ylabel("Average H-bond Length (Angstroms)")

    # making scatter plot for non-averaged h-bond distances for each sample
    for key in keys:
        distances = non_averaged_distances.get(key, [])
        x_values = [key] * len(distances)
        ax.scatter(x_values, distances, color='red', marker='o', alpha=0.7)

    plt.title(f"Average Hydrogen Bond Length between hTERT residues {residues} and hTR")
    plt.show()


def main():
    # PDB structures of wildtype and mutants
    pdb_WT = "7bg9.pdb"
    pdb_K1050E = "K1050E.pdb"
    pdb_L557P = "L557P.pdb"

    # finding the ligand residues in the samples
    WT_ligand = find_ligand_residue_range(pdb_WT)
    K1050E_ligand = find_ligand_residue_range(pdb_K1050E)
    L557P_ligand = find_ligand_residue_range(pdb_L557P)

    # determining ligand residue range in the samples
    WT_ligand_residue_range = (WT_ligand[0], WT_ligand[1])
    K1050E_ligand_residue_range = (K1050E_ligand[0], K1050E_ligand[1])
    L557P_ligand_residue_range = (L557P_ligand[0], L557P_ligand[1])

    # protein residue ranges
    residue_range1 = (482, 488)
    residue_range2 = (1050, 1050)
    residue_range3 = (11, 1132)

    # calculate hydrogen bond distances for each sample
    WT_h_bonds1 = calculate_h_bonds(pdb_WT, WT_ligand_residue_range, residue_range1)
    K1050E_h_bonds1 = calculate_h_bonds(pdb_K1050E, K1050E_ligand_residue_range, residue_range1)
    L557P_h_bonds1 = calculate_h_bonds(pdb_L557P, L557P_ligand_residue_range, residue_range1)

    WT_h_bonds2 = calculate_h_bonds(pdb_WT, WT_ligand_residue_range, residue_range2)
    K1050E_h_bonds2 = calculate_h_bonds(pdb_K1050E, K1050E_ligand_residue_range, residue_range2)
    L557P_h_bonds2 = calculate_h_bonds(pdb_L557P, L557P_ligand_residue_range, residue_range2)

    WT_h_bonds3 = calculate_h_bonds(pdb_WT, WT_ligand_residue_range, residue_range3)
    K1050E_h_bonds3 = calculate_h_bonds(pdb_K1050E, K1050E_ligand_residue_range, residue_range3)
    L557P_h_bonds3 = calculate_h_bonds(pdb_L557P, L557P_ligand_residue_range, residue_range3)

    # calculate average h bond length for each sample
    WT_h_bonds_dist1 = avg_h_bond_dist(WT_h_bonds1)
    K1050E_h_bonds_dist1 = avg_h_bond_dist(K1050E_h_bonds1)
    L557P_h_bonds_dist1 = avg_h_bond_dist(L557P_h_bonds1)

    WT_h_bonds_dist2 = avg_h_bond_dist(WT_h_bonds2)
    K1050E_h_bonds_dist2 = avg_h_bond_dist(K1050E_h_bonds2)
    L557P_h_bonds_dist2 = avg_h_bond_dist(L557P_h_bonds2)

    WT_h_bonds_dist3 = avg_h_bond_dist(WT_h_bonds3)
    K1050E_h_bonds_dist3 = avg_h_bond_dist(K1050E_h_bonds3)
    L557P_h_bonds_dist3 = avg_h_bond_dist(L557P_h_bonds3)

    # making dictionaries of results avg to plot
    h_bond_avg_results1 = {"WT": WT_h_bonds_dist1, "L557P": L557P_h_bonds_dist1, "K1050E": K1050E_h_bonds_dist1}
    h_bond_avg_results2 = {"WT": WT_h_bonds_dist2, "L557P": L557P_h_bonds_dist2, "K1050E": K1050E_h_bonds_dist2}
    h_bond_avg_results3 = {"WT": WT_h_bonds_dist3, "L557P": L557P_h_bonds_dist3, "K1050E": K1050E_h_bonds_dist3}

    # making dictionaries of non-averaged h bond distances to plot
    h_bond_results1 = {"WT": WT_h_bonds1, "L557P": L557P_h_bonds1, "K1050E": K1050E_h_bonds1}
    h_bond_results2 = {"WT": WT_h_bonds2, "L557P": L557P_h_bonds2, "K1050E": K1050E_h_bonds2}
    h_bond_results3 = {"WT": WT_h_bonds3, "L557P": L557P_h_bonds3, "K1050E": K1050E_h_bonds3}

    # making scatter plot/bar graph visualizations
    h_bond_bar_graph(h_bond_avg_results1, "482-488", h_bond_results1)
    h_bond_bar_graph(h_bond_avg_results2, "1050", h_bond_results2)
    h_bond_bar_graph(h_bond_avg_results3, "11-1132", h_bond_results3)


if __name__ == "__main__":
    main()
