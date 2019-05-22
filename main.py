import numpy as np
import scipy.special
from matplotlib import pyplot as plt
import copy

nt = [15, 12, 10, 10, 10, 40, 40]
p = 1./2

def L(stock, control, demand):
    return 2*min(stock+control, demand) - control


def solve(K=np.zeros(51), nt=[15, 12, 10, 10, 10, 40, 40], p=1./2):
    V = np.zeros((51, 8)) - float('inf')
    V[:, -1] = K
    optimal_controls = np.zeros((51, 7))
    for t in range(6, -1, -1):
        for s in range(51):
            for control in range(0, min(10, 50-s)+1):
                v_control = 0
                for d in range(nt[t]):
                    v_control += scipy.special.binom(nt[t], d)*p**d * (1-p)**(nt[t]-d) * (L(s, control, d) + V[s+control-min(d, s+control), t+1])
                if v_control > V[s, t]:
                    V[s, t] = v_control
                    optimal_controls[s, t] = control
    return V[:, 0], optimal_controls

def plot_solution():
    bellman_values = solve()[0]
    plt.plot(list(range(51)), bellman_values)
    plt.grid()
    plt.title('Values of the optimization problem')
    plt.xlabel('s0')
    plt.ylabel('Expectation')
    plt.show()


def buy_initial_stock(prices=[0.75, 1, 1.25]):
    bellman_values = solve()[0]
    i = 1
    for price in prices:
        values = bellman_values
        for k in range(51):
            values[k] -= k*price
        plt.subplot(int('13'+str(i)))
        plt.plot(list(range(51)), values)
        plt.grid()
        plt.title("Price: "+str(price)+ '\n' + "Optimal initial stock: "+str(np.argmax(values)))
        i += 1
    plt.tight_layout()
    plt.show()


def monte_carlo_simulation(s0, u, nb_iter=1000):
    gains = []
    for iteration in range(nb_iter):
        s = s0
        gain = 0
        for t in range(7):
            u_eff = min(u, 50-s)
            s = s+u_eff
            d = np.random.binomial(nt[t], p)
            gain += 2*min(s, d) - u_eff
            s = max(s-d, 0)
        gains.append(gain)
    return 1./nb_iter*np.sum(gains)
        
    
def plot_constant_u_montecarlo(s0=20):
    expected_gain = []
    for u in range(0, 51):
        expected_gain.append(monte_carlo_simulation(s0, u))
    plt.plot(list(range(51)), expected_gain)
    plt.grid()
    plt.xlabel("u")
    plt.ylabel("Expectation")
    plt.title("Values of the optimization problem with constant control"+'\n'+"Optimal constant control: "+str(np.argmax(expected_gain)))
    plt.show()

def monte_carlo_verification(s0=20, nb_iter=1000):
    V, optimal_controls = solve()
    gains = []
    for iteration in range(nb_iter):
        s = s0
        gain = 0
        for t in range(7):
            u_eff = optimal_controls[int(s), t]
            s = s+u_eff
            d = np.random.binomial(nt[t], p)
            gain += 2*min(s, d) - u_eff
            s = max(s-d, 0)
        gains.append(gain)
    expectation = 1./nb_iter*np.sum(gains)
    print(expectation, V[s0])
    return abs(expectation - V[s0])

def compare_with_selling_final_stock():
    K_selling_final_stock = np.arange(51)
    bellman_remaining_cost = solve(K=K_selling_final_stock)[0]
    print(K_selling_final_stock)
    bellman = solve()[0]
    plt.plot(np.arange(51), bellman_remaining_cost)
    plt.plot(np.arange(51), bellman, linestyle='dashed')
    plt.grid()
    plt.legend(('selling final stock', 'keeping final stock'))
    plt.xlabel('s0')
    plt.ylabel('Expectation')
    plt.show()

def solve_on_two_weeks(nt1=[15, 12, 10, 10, 10, 40, 40], nt2=[9, 12, 8, 20, 10, 40, 40]):
    values = solve(K=solve(nt=nt1)[0])[0]
    plt.plot(np.arange(51), values)
    plt.xlabel("s0")
    plt.grid()
    plt.ylabel("Expectation")
    plt.show()

if __name__=='__main__':
    plot_solution()
    buy_initial_stock()
    print(monte_carlo_simulation(20, 5))
    plot_constant_u_montecarlo()
    print(monte_carlo_verification())
    compare_with_selling_final_stock()
    solve_on_two_weeks()