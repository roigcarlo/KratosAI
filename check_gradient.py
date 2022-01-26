import numpy as np
import matplotlib.pyplot as plt

''' Function to be evaluated '''
def f(q):
    return q**3+3*q**2

''' First derivative of f(q) '''
def df(q):
    return 3*q**2+6*q

''' How to compute x axis values '''
def xAxis(q,dq,ε):
    return np.log(ε)

''' How to compute y axis values '''
def yAxis(q,dq,ε):
    return np.log(f(q+ε*dq) - f(q) - df(q)*ε*dq)

''' Calculate Slope '''
def slope(x,y):
    return np.polyfit(x, y, 1)[0]

''' Check that gradient slope is correct '''
def main():
    # Get dq something small
    dq = 1e-6

    # Get a range of ε values to evaluate the gradient at
    ε  = np.linspace(1,10,10) 

    # Evaluate at different points
    for q in np.linspace(1,10,10):
        
        # Get the x,y axis values for the plot
        x  = xAxis(q,dq,ε)
        y  = yAxis(q,dq,ε)

        # Check the slope
        s = slope(x,y)

        # Plot
        plt.plot(xAxis(q,dq,ε), yAxis(q,dq,ε), label=f'at q={q}, slope={s:0.2f}')
    
    # plt.rcParams.update({
    #     "text.usetex": True
    # })

    plt.title('$f(q)=||f(q+\\epsilon{}dq)-f(q)-\\nabla{}f|_{q}\\epsilon{}dq||$')
    plt.legend(loc="upper left")
    plt.xlabel('$log(\\epsilon{})$')
    plt.ylabel('$log(f(q))$')
    plt.show()

if __name__ == "__main__":
    main()