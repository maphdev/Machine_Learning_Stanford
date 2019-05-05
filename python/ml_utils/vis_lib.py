import matplotlib.pyplot as plt

# Plot two class with two same features
def plot2Class(A, B, labelA, labelB, labelx, labely):
    plt.figure(figsize=(10,8))
    plt.plot(A[:,1],A[:,2],'c+',label=labelA)
    plt.plot(B[:,1],B[:,2],'yo',label=labelB)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.legend()
    plt.grid(True)
