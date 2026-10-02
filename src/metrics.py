import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter


def norm_gini_coeff(predicted_probs, values): 
    predicted_probs = np.asarray(predicted_probs)
    values = np.asarray(values)
    order = np.argsort(predicted_probs)[::-1] 
    sorted_labels = values[order]
    n = len(predicted_probs)
    positives = 0
    x = np.arange(0,1, 1/n)
    true_positives = sum(k == 1 for k in values)
    L,L_perf = np.zeros(n+1), np.zeros(n+1)
    
    for i in range(1,n+1):
        if sorted_labels[i-1]==1:
            positives+=1
        L[i] = positives/true_positives
        L_perf[i] = min(i,true_positives)/true_positives
    
    area, area_perf = np.trapezoid(L, x=x), np.trapezoid(L_perf, x=x)

    
    G = area-.5
    G_perf = area_perf-.5
    return G/G_perf 

    

def lorenz_plot(predicted_probs, values):
    predicted_probs = np.asarray(predicted_probs)
    values = np.asarray(values)

    order = np.argsort(predicted_probs)[::-1]
    sorted_labels = values[order]

    n = len(values)
    true_positives = np.sum(values == 1)
    x = np.linspace(0, 1, n + 1)
    L = np.zeros(n + 1)
    L_perf = np.zeros(n + 1)

    positives = 0

    for i in range(1, n + 1):
        if sorted_labels[i - 1] == 1:
            positives += 1

        L[i] = positives / true_positives
        L_perf[i] = min(i, true_positives) / true_positives

    area = np.trapezoid(L, x=x)
    area_perf = np.trapezoid(L_perf, x=x)

    G = area - 0.5
    G_perf = area_perf - 0.5
    normalized_gini = G / G_perf

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        x, x,
        linestyle="--",
        linewidth=2,
        label="Tilfeldig modell"
    )

    ax.plot(
        x, L,
        marker="o",
        linewidth=2.5,
        label="Modell"
    )

    ax.plot(
        x, L_perf,
        linestyle=":",
        linewidth=2.5,
        label="Perfekt modell"
    )

    
    ax.fill_between(
        x,
        x,
        L,
        alpha=0.15,
        label="Gini-område"
    )

    ax.set_xlabel("Andel observasjoner sjekket", fontsize=12)
    ax.set_ylabel("Andel positive funnet", fontsize=12)

    ax.set_title(
        f"Lorenz-kurve\nNormalized Gini = {normalized_gini:.3f}",
        fontsize=14
    )

    
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
