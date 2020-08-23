import numpy as np
import matplotlib.pyplot as plt

def Gaussian(x, mu, strength, sigma):
    "Produces a Gaussian curve"
    #print("x:", x,)
    #print( "mu:", mu, "strength:", strength, "sigma:", sigma)
    #print(strength / (sigma*np.sqrt(2*np.pi)))
    #bandshape = (strength / (sigma*np.sqrt(2*np.pi)))  * np.exp(-1*((x-mu))**2/(2*(sigma**2)))
    bandshape = (strength)  * np.exp(-1*((x-mu))**2/(2*(sigma**2)))
    return bandshape


#contourplot
#Takes dictionary of (X,Y): energy entries
def contourplot(surfacedictionary, label='Label',finalunit=None):
    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcalpermol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    
    e=[]
    coords=[]
    x_c=[]
    y_c=[]
    for i in surfacedictionary:
        coords.append(i)
        x_c.append(i[0])
        y_c.append(i[1])
        e.append(surfacedictionary[i])
    
    #X and Y ranges
    x = sorted(set(x_c))
    y = sorted(set(y_c))

    #List of energies and relenergies here
    refenergy=float(min(e))
    rele=[]
    for numb in e:
        rele.append((numb-refenergy)*conversionfactor[finalunit])

    #Creating structured array
    st=[]
    for ind in range(0,len(x_c)):
        st.append((x_c[ind], y_c[ind], rele[ind]))
    rele_new=[]
    curr=[]
    for yitem in y:
        for xitem in x:
            for nz in range(0,len(st)):
                if xitem in st[nz] and yitem in st[nz]:
                    curr.append(st[nz][2])
                    energy=st[nz][2]
        rele_new.append(curr)
        curr=[]

    #Now we create contour plot
    X, Y = np.meshgrid(x, y)
    cp=plt.contourf(X, Y, rele_new, 50, alpha=.75, cmap='viridis')
    C = plt.contour(X, Y, rele_new, 50, colors='black')
    plt.colorbar(cp)
    plt.xlabel('Coord X')
    plt.ylabel('Coord Y')
    plt.savefig('Surface{}.png'.format(label), format='png', dpi=200)
    print("Created PNG file: Surface{}.png".format(label))