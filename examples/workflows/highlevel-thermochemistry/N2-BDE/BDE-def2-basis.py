from ash import *

#Script to calculate Bond dissociation energy of a simple molecule
#CCSD(T)-CBS with multiple extrapolations of def2 family

#Define molecular fragment from XYZ-file
N2=Fragment(xyzfile='n2.xyz', readchargemult=True)
#Define atomic species
N=Fragment(elems=['N'], coords=[[0.0,0.0,0.0]], charge=0, mult=4)

#Create a list of fragments and define the stoichiometry
specieslist=[N2, N]
stoichiometry=[-1,2]

#Where to store the results
resultdict={}

blocks="""
%maxcore 6000
"""

#For loop that iterates over something: here different basis sets
for basis in ['def2-SVP', 'def2-TZVPP', 'def2-QZVPP']:
    #Define theory level
    cc = ORCATheory(orcasimpleinput="! CCSD(T) {} tightscf".format(basis), orcablocks=blocks)
    #Set charge/mult of theory for N2 and run singlepoint
    cc.charge=N2.charge; cc.mult=N2.mult
    e_n2 = Singlepoint(theory=cc, fragment=N2)
    cc.cleanup()
    #Set charge/mult of theory for N and run singlepoint
    cc.charge=N.charge; cc.mult=N.mult
    e_n = Singlepoint(theory=cc, fragment=N)
    cc.cleanup()
    #Store final energies of fragment in dictionary
    resultdict['def2_'+str(basis)]=[e_n2,e_n]
    #Print BDE for each basis
    ReactionEnergy(stoichiometry=stoichiometry, label=basis, list_of_fragments=specieslist, list_of_energies=[e_n2,e_n])

print("\n\n")
#Print final results
for basis,energies in resultdict.items():
    ReactionEnergy(stoichiometry=stoichiometry, label=basis, list_of_fragments=specieslist, list_of_energies=energies)
